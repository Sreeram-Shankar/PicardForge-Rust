//defines the SDIRK step with Gauss-Seidel relaxation
pub fn step_sdirk<F>(f: F, t: f64, y: &[f64], h: f64,
                     a: &[Vec<f64>], b: &[f64], c: &[f64],
                     s: usize, n: usize, y_next: &mut [f64],
                     sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //initializes all stages with the initial guess y
    let mut y_stages: Vec<Vec<f64>> = (0..s).map(|_| y.to_vec()).collect();
    let mut y_old: Vec<Vec<f64>> = vec![vec![0.0; n]; s];
    let mut rhs = vec![0.0; n];
    let mut fval = vec![0.0; n];
    
    //implements Gauss-Seidel relaxation
    for _k in 0..sweeps {
        for i in 0..s {
            y_old[i] = y_stages[i].clone();
        }
        for i in 0..s {
            rhs.fill(0.0);
            for j in 0..s {
                f(t + c[j] * h, &y_stages[j], &mut fval);
                for idx in 0..n {
                    rhs[idx] += a[i][j] * fval[idx];
                }
            }
            for idx in 0..n {
                y_stages[i][idx] = y[idx] + h * rhs[idx];
            }
        }
        
        //computes L2 norm of all stage differences
        let mut diff_norm = 0.0;
        for i in 0..s {
            for idx in 0..n {
                let diff = y_stages[i][idx] - y_old[i][idx];
                diff_norm += diff * diff;
            }
        }
        diff_norm = diff_norm.sqrt();
        if diff_norm < tol {
            break;
        }
    }
    
    //computes the final state update
    y_next.copy_from_slice(y);
    for i in 0..s {
        f(t + c[i] * h, &y_stages[i], &mut fval);
        for idx in 0..n {
            y_next[idx] += h * b[i] * fval[idx];
        }
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK2
pub fn solve_sdirk2<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                       h: f64, n: usize, yout: &mut [Vec<f64>],
                       tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
    let a = vec![
        vec![gamma, 0.0],
        vec![1.0 - gamma, gamma],
    ];
    let b = vec![1.0 - gamma, gamma];
    let c = vec![gamma, 1.0];
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    let mut y = y0.to_vec();
    
    for k in 0..nsteps {
        step_sdirk(&f, tgrid[k], &y, h, &a, &b, &c, 2, n, &mut yout[k+1], sweeps, tol);
        y = yout[k+1].clone();
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK3
pub fn solve_sdirk3<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                       h: f64, n: usize, yout: &mut [Vec<f64>],
                       tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let gamma = 0.435866521508459;
    let a = vec![
        vec![gamma, 0.0, 0.0],
        vec![0.2820667395, gamma, 0.0],
        vec![1.208496649, -0.644363171, gamma],
    ];
    let b = vec![1.208496649, -0.644363171, gamma];
    let c = vec![gamma, 0.7179332605, 1.0];
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    let mut y = y0.to_vec();
    
    for k in 0..nsteps {
        step_sdirk(&f, tgrid[k], &y, h, &a, &b, &c, 3, n, &mut yout[k+1], sweeps, tol);
        y = yout[k+1].clone();
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation SDIRK4
pub fn solve_sdirk4<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                       h: f64, n: usize, yout: &mut [Vec<f64>],
                       tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let gamma = 0.572816062482135;
    let a = vec![
        vec![gamma, 0.0, 0.0, 0.0],
        vec![-0.6557110092, gamma, 0.0, 0.0],
        vec![0.757184241, 0.237758128, gamma, 0.0],
        vec![0.155416858, 0.701913790, 0.142669351, gamma],
    ];
    let b = vec![0.155416858, 0.701913790, 0.142669351, gamma];
    let c = vec![gamma, 0.344, 0.995, 1.0];
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    let mut y = y0.to_vec();
    
    for k in 0..nsteps {
        step_sdirk(&f, tgrid[k], &y, h, &a, &b, &c, 4, n, &mut yout[k+1], sweeps, tol);
        y = yout[k+1].clone();
    }
}