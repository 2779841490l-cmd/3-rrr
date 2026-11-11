# DeepSeek R1 Pure Mathematical Derivation Prompt (< 350 Words)

> **Usage Rule (Compute Economy)**: Feed ONLY this file to **DeepSeek R1** (zero code files, zero logs). Output goes to `docs/deepseek_r1_derivation.md` Section IV.

## Problem Statement (IEEE T-MECH / TIM Formal Proofs)

Consider a planar 2-DOF parallel linkage driven by two rotational servos $\mathbf{u} = [u_x, u_y]^T \in \mathcal{U} \subset \mathbb{R}^2$ with asymmetric backlash width $\mathbf{b} = [b_x, b_y]^T$ and nonlinear forward kinematics $\mathbf{x} = \mathbf{f}(\mathbf{u}) = [x, y]^T \in \Omega \subset \mathbb{R}^2$, where the Jacobian $J(\mathbf{u}) = \partial \mathbf{f}/\partial \mathbf{u}$ has strong cross-axis coupling ($J_{12} \neq 0, J_{21} \neq 0$), causing parasitic vertical jump $\delta y(x)$ during horizontal motion and horizontal jump $\delta x(y)$ during vertical motion.

Please derive rigorous mathematical proofs (in bilingual Chinese/English format for IEEE Transactions) for the following three theorems:

1. **Theorem 5 (Optimal Origin Shift & Maximum Inscribed Rectangle Theorem)**:
   Given the convex hull $\mathcal{H} = \text{conv}(\Omega) = \{\mathbf{x} \in \mathbb{R}^2 : A\mathbf{x} + \mathbf{c} \le \mathbf{0}\}$ of empirical reachable optical counts, formulate and solve the convex optimization problem for the shifted geometric center $\mathbf{x}_0^* = [o_x^*, o_y^*]^T$ and maximum half-widths $(W^*, H^*)$ maximizing the aspect-regularized area $\mathcal{J}(o_x, o_y, W, H) = W\cdot H$ subject to $\mathbf{x}_0^* + [\pm W, \pm H]^T \in \mathcal{H}_{-\delta}$.

2. **Theorem 6 (Cross-Axis Parasitic Jump Elimination via Bilinear Lattice Iterative Contour Learning)**:
   Let the inverse decoupling manifold on $\mathcal{R} = [-W^*, W^*] \times [-H^*, H^*]$ be parameterized by
   $$\mathbf{u}^{(k)}(x, y, \boldsymbol{\sigma}) = \mathbf{u}_{\text{kin}}(x + o_x^*, y + o_y^*) + \sum_{m,n} \phi_{mn}(x, y)\,\boldsymbol{\theta}_{mn}^{(k)} + \text{diag}(\boldsymbol{\sigma})\mathbf{b}$$
   where $\phi_{mn}(x,y)$ are bilinear/B-spline basis functions on an $M \times M$ lattice and $\boldsymbol{\sigma} = [\text{sign}(\dot{u}_x), \text{sign}(\dot{u}_y)]^T$. Prove that under the spatial kernel update law along the rectangle perimeter $\partial \mathcal{R}$ and crosshairs:
   $$\boldsymbol{\theta}_{mn}^{(k+1)} = \boldsymbol{\theta}_{mn}^{(k)} + \gamma \frac{\sum_p w_{mn}(\mathbf{x}_p) J(\mathbf{x}_p)^{-1} \mathbf{e}_p^{(k)}}{\sum_p w_{mn}(\mathbf{x}_p)}$$
   the cross-axis contour jump $\|\delta y^{(k)}\|_\infty, \|\delta x^{(k)}\|_\infty \to 0$ exponentially if $0 < \gamma < 2 / \rho(J^{-1}\hat{J})$.
