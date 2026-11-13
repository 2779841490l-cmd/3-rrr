/**
 * =============================================================================
 * BearPi-Pico H3863 (HiSilicon WS63) - 4-Axis Ultra-Precision Servo Controller
 * =============================================================================
 * 
 * Hardware Pin Mapping:
 *   - Servo 1 (X-Axis  / 左下水平推拉舵机): GPIO 2 (Pin Mode 0) -> 黄色信号线接 GPIO 2
 *   - Servo 2 (Y1-Axis / 右上垂直主提舵机): GPIO 5 (Pin Mode 4) -> 黄色信号线接 GPIO 5
 *   - Servo 3 (Y2-Axis / 右下辅助平衡舵机): GPIO 7 (Pin Mode 0) -> 黄色信号线接 GPIO 7
 *   - Servo 4 (Aux     / 备用/左上扩展舵机): GPIO 8 (Pin Mode 0) -> 黄色信号线接 GPIO 8
 * 
 * Electrical Requirements:
 *   - 棕色/黑色线: 全部接 GND（单片机与外接动力电源必须共地）
 *   - 红色线: 接外部 5V~6V (5A+) 独立舵机供电电源（严禁接单片机 3.3V）
 *   - 黄色/橙色线: 信号线依次接 GPIO 2, GPIO 5, GPIO 7, GPIO 8
 * 
 * Real-Time UART Protocol (COM4, 115200 baud, 8N1):
 *   - "H <t1> <t2> <t3> <t4>\n"  -> 4-axis 0.1us pulse widths (5000~25000 => 500.0~2500.0 us)
 *   - "H <t1> <t2> <t3>\n"       -> 3-axis 0.1us backward-compatible (s4=15000)
 *   - "B <b1> <b2> <e1> <e2> <bst_us>\n" -> Hardware-timed ballistic pulse (0.1us widths) for bst_us, then auto-switch to eq
 *   - "P <s1> <s2> <s3> <s4>\n"  -> Legacy 4-axis us (int or 1-dec float, 500~2500 us)
 *   - "P <s1> <s2> <s3>\n"       -> Legacy 3-axis us (auto s4=1500.0)
 *   - "C\n" or "c\n"             -> Neutral reset (1510.0, 1545.0, 1455.0, 1500.0 us)
 *   - "V\n" or "v\n"             -> Query firmware capability ("FW_CAP: SUBUS_V2 ... B_CMD")
 * =============================================================================
 */

#include "pinctrl.h"
#include "common_def.h"
#include "soc_osal.h"
#include "gpio.h"
#include "hal_gpio.h"
#include "uart.h"
#include "tcxo.h"
#include "tcxo_porting.h"
#include "watchdog.h"
#include "app_init.h"

// -----------------------------------------------------------------------------
// GPIO Pin Definitions
// -----------------------------------------------------------------------------
#define SERVO_PIN_1             2   // GPIO 2  (X轴: 水平推拉)
#define SERVO_PIN_1_MODE        0   // Mode 0 = GPIO_02

#define SERVO_PIN_2             5   // GPIO 5  (Y1轴: 垂直主提)
#define SERVO_PIN_2_MODE        4   // Mode 4 = GPIO_05

#define SERVO_PIN_3             7   // GPIO 7  (Y2轴: 垂直副提/反装)
#define SERVO_PIN_3_MODE        0   // Mode 0 = GPIO_07

#define SERVO_PIN_4             8   // GPIO 8  (Aux/备用/第4舵机)
#define SERVO_PIN_4_MODE        0   // Mode 0 = GPIO_08

#define SERVO_TASK_STACK_SIZE   0x1000
#define SERVO_TASK_PRIO         17

// -----------------------------------------------------------------------------
// Neutral Default Constants (in 0.1 us units; 15100 == 1510.0 us)
// -----------------------------------------------------------------------------
#define DEFAULT_UX_NEUTRAL_01US      15100
#define DEFAULT_UY1_NEUTRAL_01US     15450
#define DEFAULT_UY2_NEUTRAL_01US     14550
#define DEFAULT_S4_NEUTRAL_01US      15000
#define SERVO_MIN_01US               11500
#define SERVO_MAX_01US               18500
#define SERVO_LOW_MIN_US             4500U  // KC5188 ramp capacitor requires >= 4.0ms low recovery

static volatile uint32_t g_cmd_s1_01us = DEFAULT_UX_NEUTRAL_01US;
static volatile uint32_t g_cmd_s2_01us = DEFAULT_UY1_NEUTRAL_01US;
static volatile uint32_t g_cmd_s3_01us = DEFAULT_UY2_NEUTRAL_01US;
static volatile uint32_t g_cmd_s4_01us = DEFAULT_S4_NEUTRAL_01US;
static volatile uint32_t g_pwm_period_us = 10000;  // Default 100Hz (10,000us) safe steady-state; switchable via "F <us>\n"
static volatile uint8_t  g_cmd_updated = 0;       // Event-triggered immediate pulse flag on UART RX

// Hardware-timed ballistic-to-equilibrium state ("B <b1> <b2> <e1> <e2> <bst_us>\n")
static volatile uint8_t  g_bst_active = 0;
static volatile uint8_t  g_bst_started = 0;
static volatile uint32_t g_cmd_seq = 0;
static volatile uint64_t g_bst_t_start_ticks = 0;
static volatile uint64_t g_bst_dur_ticks = 0;
static volatile uint32_t g_bst_s1_01us = DEFAULT_UX_NEUTRAL_01US;
static volatile uint32_t g_bst_s2_01us = DEFAULT_UY1_NEUTRAL_01US;
static volatile uint32_t g_bst_s3_01us = DEFAULT_UY2_NEUTRAL_01US;
static volatile uint32_t g_bst_s4_01us = DEFAULT_S4_NEUTRAL_01US;
static volatile uint32_t g_eq_s1_01us  = DEFAULT_UX_NEUTRAL_01US;
static volatile uint32_t g_eq_s2_01us  = DEFAULT_UY1_NEUTRAL_01US;
static volatile uint32_t g_eq_s3_01us  = DEFAULT_UY2_NEUTRAL_01US;
static volatile uint32_t g_eq_s4_01us  = DEFAULT_S4_NEUTRAL_01US;

// Non-blocking UART buffer
static char g_line_buf[128];
static uint32_t g_line_idx = 0;

static inline uint32_t clamp_servo_01us(int v)
{
    if (v < SERVO_MIN_01US) return SERVO_MIN_01US;
    if (v > SERVO_MAX_01US) return SERVO_MAX_01US;
    return (uint32_t)v;
}

// -----------------------------------------------------------------------------
// UART Command Parser
// -----------------------------------------------------------------------------
static int parse_us_to_01us(const char **pp)
{
    const char *p = *pp;
    int whole = 0;
    int tenths = 0;
    while (*p >= '0' && *p <= '9') {
        whole = whole * 10 + (*p - '0');
        p++;
    }
    if (*p == '.') {
        p++;
        if (*p >= '0' && *p <= '9') {
            tenths = (*p - '0');
            p++;
            if (*p >= '5' && *p <= '9' && tenths < 9) {
                tenths++;
            }
            while (*p >= '0' && *p <= '9') {
                p++;
            }
        }
    }
    *pp = p;
    return whole * 10 + tenths;
}

static void parse_uart_line(const char *line)
{
    if (line[0] == 'H' || line[0] == 'h') {
        int val[4] = {0, 0, 0, 0};
        int count = 0;
        const char *p = line + 1;

        while (*p && count < 4) {
            while (*p == ' ' || *p == ',' || *p == '\t') {
                p++;
            }
            if (*p >= '0' && *p <= '9') {
                while (*p >= '0' && *p <= '9') {
                    val[count] = val[count] * 10 + (*p - '0');
                    p++;
                }
                count++;
            } else if (*p == '\r' || *p == '\n' || *p == '\0') {
                break;
            } else {
                p++;
            }
        }

        if (count >= 3) {
            unsigned int irq = osal_irq_lock();
            g_bst_active = 0;
            g_bst_started = 0;
            if (val[0] >= SERVO_MIN_01US && val[0] <= SERVO_MAX_01US) g_cmd_s1_01us = (uint32_t)val[0];
            if (val[1] >= SERVO_MIN_01US && val[1] <= SERVO_MAX_01US) g_cmd_s2_01us = (uint32_t)val[1];
            if (val[2] >= SERVO_MIN_01US && val[2] <= SERVO_MAX_01US) g_cmd_s3_01us = (uint32_t)val[2];
            if (count == 4 && val[3] >= SERVO_MIN_01US && val[3] <= SERVO_MAX_01US) {
                g_cmd_s4_01us = (uint32_t)val[3];
            } else if (count == 3) {
                g_cmd_s4_01us = DEFAULT_S4_NEUTRAL_01US;
            }
            g_cmd_seq++;
            g_cmd_updated = 1;
            osal_irq_restore(irq);
        }
    } else if (line[0] == 'B' || line[0] == 'b') {
        int val[5] = {0, 0, 0, 0, 0};
        int count = 0;
        const char *p = line + 1;

        while (*p && count < 5) {
            while (*p == ' ' || *p == ',' || *p == '\t') {
                p++;
            }
            if (*p >= '0' && *p <= '9') {
                while (*p >= '0' && *p <= '9') {
                    val[count] = val[count] * 10 + (*p - '0');
                    p++;
                }
                count++;
            } else if (*p == '\r' || *p == '\n' || *p == '\0') {
                break;
            } else {
                p++;
            }
        }

        if (count == 5) {
            uint32_t dur_us = (val[4] < 0) ? 0U : ((val[4] > 100000) ? 100000U : (uint32_t)val[4]);
            uint64_t tpu = (uint64_t)tcxo_porting_ticks_per_usec_get();
            if (tpu == 0ULL) {
                tpu = 40ULL;
            }
            uint32_t b1 = clamp_servo_01us(val[0]);
            uint32_t b2 = clamp_servo_01us(val[1]);
            uint32_t b3 = clamp_servo_01us(30000 - (int)b2);
            uint32_t e1 = clamp_servo_01us(val[2]);
            uint32_t e2 = clamp_servo_01us(val[3]);
            uint32_t e3 = clamp_servo_01us(30000 - (int)e2);

            unsigned int irq = osal_irq_lock();
            g_bst_s1_01us = b1;
            g_bst_s2_01us = b2;
            g_bst_s3_01us = b3;
            g_bst_s4_01us = DEFAULT_S4_NEUTRAL_01US;
            g_eq_s1_01us  = e1;
            g_eq_s2_01us  = e2;
            g_eq_s3_01us  = e3;
            g_eq_s4_01us  = DEFAULT_S4_NEUTRAL_01US;
            g_bst_dur_ticks = (uint64_t)dur_us * tpu;
            g_bst_started = 0;
            if (dur_us > 0U) {
                g_bst_active = 1;
                g_cmd_s1_01us = b1;
                g_cmd_s2_01us = b2;
                g_cmd_s3_01us = b3;
                g_cmd_s4_01us = DEFAULT_S4_NEUTRAL_01US;
            } else {
                g_bst_active = 0;
                g_cmd_s1_01us = e1;
                g_cmd_s2_01us = e2;
                g_cmd_s3_01us = e3;
                g_cmd_s4_01us = DEFAULT_S4_NEUTRAL_01US;
            }
            g_cmd_seq++;
            g_cmd_updated = 1;
            osal_irq_restore(irq);
        }
    } else if (line[0] == 'P' || line[0] == 'p') {
        int val_01[4] = {0, 0, 0, 0};
        int count = 0;
        const char *p = line + 1;

        while (*p && count < 4) {
            while (*p == ' ' || *p == ',' || *p == '\t') {
                p++;
            }
            if (*p >= '0' && *p <= '9') {
                val_01[count++] = parse_us_to_01us(&p);
            } else if (*p == '\r' || *p == '\n' || *p == '\0') {
                break;
            } else {
                p++;
            }
        }

        if (count == 4) {
            unsigned int irq = osal_irq_lock();
            g_bst_active = 0;
            g_bst_started = 0;
            if (val_01[0] >= SERVO_MIN_01US && val_01[0] <= SERVO_MAX_01US) g_cmd_s1_01us = (uint32_t)val_01[0];
            if (val_01[1] >= SERVO_MIN_01US && val_01[1] <= SERVO_MAX_01US) g_cmd_s2_01us = (uint32_t)val_01[1];
            if (val_01[2] >= SERVO_MIN_01US && val_01[2] <= SERVO_MAX_01US) g_cmd_s3_01us = (uint32_t)val_01[2];
            if (val_01[3] >= SERVO_MIN_01US && val_01[3] <= SERVO_MAX_01US) g_cmd_s4_01us = (uint32_t)val_01[3];
            g_cmd_seq++;
            g_cmd_updated = 1;
            osal_irq_restore(irq);
        } else if (count == 3) {
            unsigned int irq = osal_irq_lock();
            g_bst_active = 0;
            g_bst_started = 0;
            if (val_01[0] >= SERVO_MIN_01US && val_01[0] <= SERVO_MAX_01US) g_cmd_s1_01us = (uint32_t)val_01[0];
            if (val_01[1] >= SERVO_MIN_01US && val_01[1] <= SERVO_MAX_01US) g_cmd_s2_01us = (uint32_t)val_01[1];
            if (val_01[2] >= SERVO_MIN_01US && val_01[2] <= SERVO_MAX_01US) g_cmd_s3_01us = (uint32_t)val_01[2];
            g_cmd_s4_01us = DEFAULT_S4_NEUTRAL_01US;
            g_cmd_seq++;
            g_cmd_updated = 1;
            osal_irq_restore(irq);
        }
    } else if (line[0] == 'C' || line[0] == 'c') {
        // Center all to neutral
        unsigned int irq = osal_irq_lock();
        g_bst_active = 0;
        g_bst_started = 0;
        g_cmd_s1_01us = DEFAULT_UX_NEUTRAL_01US;
        g_cmd_s2_01us = DEFAULT_UY1_NEUTRAL_01US;
        g_cmd_s3_01us = DEFAULT_UY2_NEUTRAL_01US;
        g_cmd_s4_01us = DEFAULT_S4_NEUTRAL_01US;
        g_cmd_seq++;
        g_cmd_updated = 1;
        osal_irq_restore(irq);
    } else if (line[0] == 'V' || line[0] == 'v') {
        osal_printk("FW_CAP: SUBUS_V2 TCXO_%uMHz 0.1us 100Hz Tlow4500 B_CMD\r\n",
                    (unsigned int)tcxo_porting_ticks_per_usec_get());
    } else if (line[0] == 'F' || line[0] == 'f') {
        // Dynamic PWM Period / Frequency command: "F <period_us>\n"
        // e.g. "F 10000\n" -> 100Hz steady-state, "F 20000\n" -> 50Hz standard
        int period = 0;
        const char *p = line + 1;
        while (*p == ' ' || *p == '\t') p++;
        while (*p >= '0' && *p <= '9') {
            period = period * 10 + (*p - '0');
            p++;
        }
        if (period >= 6500 && period <= 50000) {
            g_pwm_period_us = (uint32_t)period;
        }
    }
}

static void servo_uart_rx_callback(const void *buffer, uint16_t length, bool error)
{
    unused(error);
    if (buffer == NULL || length == 0) {
        return;
    }
    const char *data = (const char *)buffer;
    for (uint16_t i = 0; i < length; i++) {
        char c = data[i];
        if (c == '\r' || c == '\n') {
            if (g_line_idx > 0) {
                g_line_buf[g_line_idx] = '\0';
                parse_uart_line(g_line_buf);
                g_line_idx = 0;
            }
        } else if (g_line_idx < sizeof(g_line_buf) - 1) {
            g_line_buf[g_line_idx++] = c;
        } else {
            g_line_idx = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// Hardware Initialization
// -----------------------------------------------------------------------------
static void servo_hardware_init(void)
{
    uapi_pin_init();
    uapi_gpio_init();
    uapi_tcxo_init();

    // 1. Configure Pin Multiplexing Mode
    uapi_pin_set_mode(SERVO_PIN_1, SERVO_PIN_1_MODE);
    uapi_pin_set_mode(SERVO_PIN_2, SERVO_PIN_2_MODE);
    uapi_pin_set_mode(SERVO_PIN_3, SERVO_PIN_3_MODE);
    uapi_pin_set_mode(SERVO_PIN_4, SERVO_PIN_4_MODE);

    // 2. Set Strong Drive Strength to ensure clean 50Hz edges
    uapi_pin_set_ds(SERVO_PIN_1, PIN_DS_4);
    uapi_pin_set_ds(SERVO_PIN_2, PIN_DS_4);
    uapi_pin_set_ds(SERVO_PIN_3, PIN_DS_4);
    uapi_pin_set_ds(SERVO_PIN_4, PIN_DS_4);

    // 3. Set Directions to Output
    uapi_gpio_set_dir(SERVO_PIN_1, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_dir(SERVO_PIN_2, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_dir(SERVO_PIN_3, GPIO_DIRECTION_OUTPUT);
    uapi_gpio_set_dir(SERVO_PIN_4, GPIO_DIRECTION_OUTPUT);

    // Initial low
    uapi_gpio_set_val(SERVO_PIN_1, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(SERVO_PIN_2, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(SERVO_PIN_3, GPIO_LEVEL_LOW);
    uapi_gpio_set_val(SERVO_PIN_4, GPIO_LEVEL_LOW);

    // 4. Register UART0 RX Callback
    (void)uapi_uart_register_rx_callback(UART_BUS_0,
        UART_RX_CONDITION_FULL_OR_SUFFICIENT_DATA_OR_IDLE, 1, servo_uart_rx_callback);

    osal_printk("\r\n");
    osal_printk("============================================================\r\n");
    osal_printk("  BearPi-Pico H3863 (WS63) - 4-Axis Anti-Jitter Firmware    \r\n");
    osal_printk("  FW_CAP: SUBUS_V2 24MHz 0.1us B_CMD                        \r\n");
    osal_printk("  Channels:                                                 \r\n");
    osal_printk("    - Servo 1 (X)  : GPIO 2  [Neutral %d.%dus]              \r\n", DEFAULT_UX_NEUTRAL_01US / 10, DEFAULT_UX_NEUTRAL_01US % 10);
    osal_printk("    - Servo 2 (Y1) : GPIO 5  [Neutral %d.%dus]              \r\n", DEFAULT_UY1_NEUTRAL_01US / 10, DEFAULT_UY1_NEUTRAL_01US % 10);
    osal_printk("    - Servo 3 (Y2) : GPIO 7  [Neutral %d.%dus]              \r\n", DEFAULT_UY2_NEUTRAL_01US / 10, DEFAULT_UY2_NEUTRAL_01US % 10);
    osal_printk("    - Servo 4 (Aux): GPIO 8  [Neutral %d.%dus]              \r\n", DEFAULT_S4_NEUTRAL_01US / 10, DEFAULT_S4_NEUTRAL_01US % 10);
    osal_printk("  Timing Engine: 24MHz TCXO Atomic Sub-us (0.1us / 41.7ns)  \r\n");
    osal_printk("============================================================\r\n\r\n");
}

// -----------------------------------------------------------------------------
// Ultra-Low Jitter 4-Axis Sub-Microsecond Pulse Generator (24MHz TCXO Ticks)
// -----------------------------------------------------------------------------
typedef struct {
    uint32_t pulse_01us;
    uint32_t pin;
} PulseItem;

static void output_pulse_4axis_atomic(
    uint32_t s1_01us,
    uint32_t s2_01us,
    uint32_t s3_01us,
    uint32_t s4_01us,
    uint8_t mark_bst_start,
    uint32_t seq_snapshot
)
{
    // Clamp to [500.0, 2500.0] us safe physical limits
    if (s1_01us < SERVO_MIN_01US) s1_01us = SERVO_MIN_01US; else if (s1_01us > SERVO_MAX_01US) s1_01us = SERVO_MAX_01US;
    if (s2_01us < SERVO_MIN_01US) s2_01us = SERVO_MIN_01US; else if (s2_01us > SERVO_MAX_01US) s2_01us = SERVO_MAX_01US;
    if (s3_01us < SERVO_MIN_01US) s3_01us = SERVO_MIN_01US; else if (s3_01us > SERVO_MAX_01US) s3_01us = SERVO_MAX_01US;
    if (s4_01us < SERVO_MIN_01US) s4_01us = SERVO_MIN_01US; else if (s4_01us > SERVO_MAX_01US) s4_01us = SERVO_MAX_01US;

    PulseItem items[4] = {
        {s1_01us, SERVO_PIN_1},
        {s2_01us, SERVO_PIN_2},
        {s3_01us, SERVO_PIN_3},
        {s4_01us, SERVO_PIN_4}
    };

    // 4-element ascending bubble sort
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3 - i; j++) {
            if (items[j].pulse_01us > items[j + 1].pulse_01us) {
                PulseItem tmp = items[j];
                items[j] = items[j + 1];
                items[j + 1] = tmp;
            }
        }
    }

    // Precompute exact dynamic TCXO tick offsets (24MHz or 40MHz depending on crystal)
    uint64_t ticks_per_us = (uint64_t)tcxo_porting_ticks_per_usec_get();
    if (ticks_per_us == 0ULL) {
        ticks_per_us = 40ULL;
    }
    uint64_t ticks_rel[4];
    for (int k = 0; k < 4; k++) {
        ticks_rel[k] = ((uint64_t)items[k].pulse_01us * ticks_per_us + 5ULL) / 10ULL;
    }

    // -------------------------------------------------------------------------
    // CRITICAL SECTION START: Disable interrupts only during high-pulse phase
    // Single common t_rise reference eliminates per-channel call-overhead drift.
    // -------------------------------------------------------------------------
    unsigned int irq_status = osal_irq_lock();

    // Pull all 4 servo pins HIGH simultaneously
    uapi_gpio_set_val(SERVO_PIN_1, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(SERVO_PIN_2, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(SERVO_PIN_3, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(SERVO_PIN_4, GPIO_LEVEL_HIGH);

    uint64_t t_rise = uapi_tcxo_get_count();
    if (mark_bst_start && g_bst_active && (g_cmd_seq == seq_snapshot)) {
        g_bst_t_start_ticks = t_rise;
        g_bst_started = 1;
    }
    for (int k = 0; k < 4; k++) {
        uint64_t t_target = t_rise + ticks_rel[k];
        while (uapi_tcxo_get_count() < t_target) {
        }
        uapi_gpio_set_val(items[k].pin, GPIO_LEVEL_LOW);
    }

    // Restore interrupts immediately after high pulses end
    osal_irq_restore(irq_status);
    // -------------------------------------------------------------------------
    // CRITICAL SECTION END
    // -------------------------------------------------------------------------

    // Low-Pulse Recovery with Circuit Protection:
    // Enforce T_low_min >= 4500us so KC5188 internal ramp capacitor fully discharges before
    // any early UART "H"/"P" wakeup or hardware-timed B->eq switchover.
    uint32_t max_pulse_us = (items[3].pulse_01us + 5U) / 10U;
    uint32_t total_low_us = (g_pwm_period_us > max_pulse_us) ? (g_pwm_period_us - max_pulse_us) : SERVO_LOW_MIN_US;
    if (total_low_us < SERVO_LOW_MIN_US) {
        total_low_us = SERVO_LOW_MIN_US;
    }
    uapi_tcxo_delay_us(SERVO_LOW_MIN_US);
    uint32_t rem_us = total_low_us - SERVO_LOW_MIN_US;
    while (rem_us > 0U && !g_cmd_updated) {
        if (g_bst_active && g_bst_started &&
            ((uapi_tcxo_get_count() - g_bst_t_start_ticks) >= g_bst_dur_ticks)) {
            break;
        }
        uint32_t step = (rem_us > 50U) ? 50U : rem_us;
        uapi_tcxo_delay_us(step);
        rem_us -= step;
    }

    uapi_watchdog_kick();
}

// -----------------------------------------------------------------------------
// Main Servo Task
// -----------------------------------------------------------------------------
static void *servo_test_task(const char *arg)
{
    unused(arg);
    servo_hardware_init();

    while (1) {
        unsigned int irq = osal_irq_lock();
        if (g_bst_active && g_bst_started) {
            uint64_t elapsed_ticks = uapi_tcxo_get_count() - g_bst_t_start_ticks;
            if (elapsed_ticks >= g_bst_dur_ticks) {
                g_cmd_s1_01us = g_eq_s1_01us;
                g_cmd_s2_01us = g_eq_s2_01us;
                g_cmd_s3_01us = g_eq_s3_01us;
                g_cmd_s4_01us = g_eq_s4_01us;
                g_bst_active = 0;
                g_bst_started = 0;
            }
        }
        uint32_t s1 = g_cmd_s1_01us;
        uint32_t s2 = g_cmd_s2_01us;
        uint32_t s3 = g_cmd_s3_01us;
        uint32_t s4 = g_cmd_s4_01us;
        uint8_t mark_bst_start = (g_bst_active && !g_bst_started) ? 1U : 0U;
        uint32_t seq_snapshot = g_cmd_seq;
        g_cmd_updated = 0;
        osal_irq_restore(irq);

        output_pulse_4axis_atomic(s1, s2, s3, s4, mark_bst_start, seq_snapshot);
    }

    return NULL;
}

static void servo_test_entry(void)
{
    osal_task *task = NULL;
    osal_kthread_lock();
    task = osal_kthread_create((osal_kthread_handler)servo_test_task, 0, "ServoTask", SERVO_TASK_STACK_SIZE);
    if (task != NULL) {
        osal_kthread_set_priority(task, SERVO_TASK_PRIO);
    }
    osal_kthread_unlock();
}

app_run(servo_test_entry);
