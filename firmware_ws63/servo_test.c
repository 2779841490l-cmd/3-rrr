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
 * Real-Time UART Protocol (COM7, 115200 baud, 8N1):
 *   - "P <s1> <s2> <s3> <s4>\n"  -> 4路脉宽微秒设定 (500~2500 us)
 *   - "P <s1> <s2> <s3>\n"       -> 3路向下兼容模式 (自动补齐 s4=1500)
 *   - "C\n" 或 "c\n"             -> 立即居中锁定复位 (1510, 1545, 1455, 1500)
 * =============================================================================
 */

#include "pinctrl.h"
#include "common_def.h"
#include "soc_osal.h"
#include "gpio.h"
#include "hal_gpio.h"
#include "uart.h"
#include "tcxo.h"
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
// Neutral Default Constants (严格对齐 calib_params.json 零位标定)
// -----------------------------------------------------------------------------
#define DEFAULT_UX_NEUTRAL      1510
#define DEFAULT_UY1_NEUTRAL     1545
#define DEFAULT_UY2_NEUTRAL     1455
#define DEFAULT_S4_NEUTRAL      1500

static volatile uint32_t g_cmd_s1_us = DEFAULT_UX_NEUTRAL;
static volatile uint32_t g_cmd_s2_us = DEFAULT_UY1_NEUTRAL;
static volatile uint32_t g_cmd_s3_us = DEFAULT_UY2_NEUTRAL;
static volatile uint32_t g_cmd_s4_us = DEFAULT_S4_NEUTRAL;
static volatile uint32_t g_pwm_period_us = 5000;  // Default 200Hz (5,000us) high-frequency mode; switchable via "F <us>\n"
static volatile uint8_t  g_cmd_updated = 0;       // Event-triggered immediate pulse flag on UART RX

// Non-blocking UART buffer
static char g_line_buf[64];
static uint32_t g_line_idx = 0;

// -----------------------------------------------------------------------------
// UART Command Parser
// -----------------------------------------------------------------------------
static void parse_uart_line(const char *line)
{
    if (line[0] == 'P' || line[0] == 'p') {
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

        if (count == 4) {
            if (val[0] >= 500 && val[0] <= 2500) g_cmd_s1_us = (uint32_t)val[0];
            if (val[1] >= 500 && val[1] <= 2500) g_cmd_s2_us = (uint32_t)val[1];
            if (val[2] >= 500 && val[2] <= 2500) g_cmd_s3_us = (uint32_t)val[2];
            if (val[3] >= 500 && val[3] <= 2500) g_cmd_s4_us = (uint32_t)val[3];
            g_cmd_updated = 1;
        } else if (count == 3) {
            // 3-axis backward compatibility
            if (val[0] >= 500 && val[0] <= 2500) g_cmd_s1_us = (uint32_t)val[0];
            if (val[1] >= 500 && val[1] <= 2500) g_cmd_s2_us = (uint32_t)val[1];
            if (val[2] >= 500 && val[2] <= 2500) g_cmd_s3_us = (uint32_t)val[2];
            g_cmd_s4_us = DEFAULT_S4_NEUTRAL;
            g_cmd_updated = 1;
        }
    } else if (line[0] == 'C' || line[0] == 'c') {
        // Center all to neutral
        g_cmd_s1_us = DEFAULT_UX_NEUTRAL;
        g_cmd_s2_us = DEFAULT_UY1_NEUTRAL;
        g_cmd_s3_us = DEFAULT_UY2_NEUTRAL;
        g_cmd_s4_us = DEFAULT_S4_NEUTRAL;
        g_cmd_updated = 1;
    } else if (line[0] == 'F' || line[0] == 'f') {
        // Dynamic PWM Period / Frequency command: "F <period_us>\n"
        // e.g. "F 5000\n" -> 200Hz overclock, "F 20000\n" -> 50Hz standard
        int period = 0;
        const char *p = line + 1;
        while (*p == ' ' || *p == '\t') p++;
        while (*p >= '0' && *p <= '9') {
            period = period * 10 + (*p - '0');
            p++;
        }
        if (period >= 3000 && period <= 50000) {
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
    osal_printk("  Channels:                                                 \r\n");
    osal_printk("    - Servo 1 (X)  : GPIO 2  [Neutral %dus]                 \r\n", DEFAULT_UX_NEUTRAL);
    osal_printk("    - Servo 2 (Y1) : GPIO 5  [Neutral %dus]                 \r\n", DEFAULT_UY1_NEUTRAL);
    osal_printk("    - Servo 3 (Y2) : GPIO 7  [Neutral %dus]                 \r\n", DEFAULT_UY2_NEUTRAL);
    osal_printk("    - Servo 4 (Aux): GPIO 8  [Neutral %dus]                 \r\n", DEFAULT_S4_NEUTRAL);
    osal_printk("  Timing Engine: Atomic IRQ-Locked Pulse Generator (Zero Jitter)\r\n");
    osal_printk("============================================================\r\n\r\n");
}

// -----------------------------------------------------------------------------
// Ultra-Low Jitter 4-Axis Pulse Generator
// -----------------------------------------------------------------------------
typedef struct {
    uint32_t pulse;
    uint32_t pin;
} PulseItem;

static void output_pulse_4axis_atomic(uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4)
{
    // Clamp to [500, 2500] safe physical limits
    if (s1 < 500) s1 = 500; else if (s1 > 2500) s1 = 2500;
    if (s2 < 500) s2 = 500; else if (s2 > 2500) s2 = 2500;
    if (s3 < 500) s3 = 500; else if (s3 > 2500) s3 = 2500;
    if (s4 < 500) s4 = 500; else if (s4 > 2500) s4 = 2500;

    PulseItem items[4] = {
        {s1, SERVO_PIN_1},
        {s2, SERVO_PIN_2},
        {s3, SERVO_PIN_3},
        {s4, SERVO_PIN_4}
    };

    // 4-element ascending bubble sort
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3 - i; j++) {
            if (items[j].pulse > items[j + 1].pulse) {
                PulseItem tmp = items[j];
                items[j] = items[j + 1];
                items[j + 1] = tmp;
            }
        }
    }

    // -------------------------------------------------------------------------
    // CRITICAL SECTION START: Disable interrupts only during high-pulse phase
    // Max high pulse duration is <= 2500us (2.5ms). UART FIFO easily handles it.
    // -------------------------------------------------------------------------
    unsigned int irq_status = osal_irq_lock();

    // Pull all 4 servo pins HIGH simultaneously
    uapi_gpio_set_val(SERVO_PIN_1, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(SERVO_PIN_2, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(SERVO_PIN_3, GPIO_LEVEL_HIGH);
    uapi_gpio_set_val(SERVO_PIN_4, GPIO_LEVEL_HIGH);

    // 1st channel drop
    uapi_tcxo_delay_us(items[0].pulse);
    uapi_gpio_set_val(items[0].pin, GPIO_LEVEL_LOW);

    // 2nd channel drop
    if (items[1].pulse > items[0].pulse) {
        uapi_tcxo_delay_us(items[1].pulse - items[0].pulse);
    }
    uapi_gpio_set_val(items[1].pin, GPIO_LEVEL_LOW);

    // 3rd channel drop
    if (items[2].pulse > items[1].pulse) {
        uapi_tcxo_delay_us(items[2].pulse - items[1].pulse);
    }
    uapi_gpio_set_val(items[2].pin, GPIO_LEVEL_LOW);

    // 4th channel drop
    if (items[3].pulse > items[2].pulse) {
        uapi_tcxo_delay_us(items[3].pulse - items[2].pulse);
    }
    uapi_gpio_set_val(items[3].pin, GPIO_LEVEL_LOW);

    // Restore interrupts immediately after high pulses end
    osal_irq_restore(irq_status);
    // -------------------------------------------------------------------------
    // CRITICAL SECTION END
    // -------------------------------------------------------------------------

    // Event-Triggered Low-Pulse Recovery:
    // Enforce minimum 2500us inter-pulse gap, then poll in 50us slices so any
    // incoming UART "P" command triggers a new PWM frame within <50us (zero phase lottery).
    uint32_t max_pulse = items[3].pulse;
    if (max_pulse < g_pwm_period_us) {
        uint32_t total_low_us = g_pwm_period_us - max_pulse;
        uint32_t min_gap_us = (total_low_us > 2500) ? 2500 : total_low_us;
        uapi_tcxo_delay_us(min_gap_us);
        uint32_t rem_us = total_low_us - min_gap_us;
        while (rem_us > 0 && !g_cmd_updated) {
            uint32_t step = (rem_us > 50) ? 50 : rem_us;
            uapi_tcxo_delay_us(step);
            rem_us -= step;
        }
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
        g_cmd_updated = 0;
        // Continuous 200Hz / Event-Triggered generation holding current commanded positions
        output_pulse_4axis_atomic(g_cmd_s1_us, g_cmd_s2_us, g_cmd_s3_us, g_cmd_s4_us);
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
