//solves the nonlinear system of equations with a Gauss-Seidel relaxation AM2
pub fn solve_am2<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_next = vec![0.0; n];
    
    f(tgrid[0], &yout[0], &mut f_vals[0]);
    
    //uses BE bootstrap
    f(tgrid[1], &yout[0], &mut f_next);
    yout[1] = (0..n).map(|i| yout[0][i] + h * f_next[i]).collect();
    f(tgrid[1], &yout[1], &mut f_vals[1]);
    
    for k in 1..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_next);
            for j in 0..n {
                y[j] = yout[k][j] + h * (0.5 * f_next[j] + 0.5 * f_vals[k][j]);
            }
            let mut diff_norm = 0.0;
            for j in 0..n {
                let diff = y[j] - y_old[j];
                diff_norm += diff * diff;
            }
            diff_norm = diff_norm.sqrt();
            if diff_norm < tol {
                break;
            }
        }
        
        yout[k+1] = y.clone();
        f_vals[k+1] = f_next.clone();
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation AM3
pub fn solve_am3<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //uses AM2 to bootstrap 1 step
    let mut y_boot: Vec<Vec<f64>> = vec![vec![0.0; n]; 3];
    let mut tgrid_boot = vec![0.0; 3];
    solve_am2(&f, t0, t0+2.0*h, y0, h, n, &mut y_boot, &mut tgrid_boot, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    for i in 0..3.min(y_boot.len()) {
        yout[i] = y_boot[i].clone();
    }
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    
    //compute initial F
    for i in 0..3 {
        f(tgrid[i], &yout[i], &mut f_vals[i]);
    }
    
    //defines the main AM3 solver
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_next = vec![0.0; n];
    for k in 2..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_next);
            for j in 0..n {
                y[j] = yout[k][j] + h * ((5.0/12.0) * f_next[j] + (2.0/3.0) * f_vals[k][j] - (1.0/12.0) * f_vals[k-1][j]);
            }
            let mut diff_norm = 0.0;
            for j in 0..n {
                let diff = y[j] - y_old[j];
                diff_norm += diff * diff;
            }
            diff_norm = diff_norm.sqrt();
            if diff_norm < tol {
                break;
            }
        }
        
        yout[k+1] = y.clone();
        f_vals[k+1] = f_next.clone();
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation AM4
pub fn solve_am4<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //uses AM3 to bootstrap 2 steps
    let mut y_boot: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];
    let mut tgrid_boot = vec![0.0; 4];
    solve_am3(&f, t0, t0+3.0*h, y0, h, n, &mut y_boot, &mut tgrid_boot, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    for i in 0..4.min(y_boot.len()) {
        yout[i] = y_boot[i].clone();
    }
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    
    //compute initial F
    for i in 0..4 {
        f(tgrid[i], &yout[i], &mut f_vals[i]);
    }
    
    //defines the main AM4 solver
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_next = vec![0.0; n];
    for k in 3..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_next);
            for j in 0..n {
                y[j] = yout[k][j] + h * ((3.0/8.0) * f_next[j] + (19.0/24.0) * f_vals[k][j] -
                                          (5.0/24.0) * f_vals[k-1][j] + (1.0/24.0) * f_vals[k-2][j]);
            }
            let mut diff_norm = 0.0;
            for j in 0..n {
                let diff = y[j] - y_old[j];
                diff_norm += diff * diff;
            }
            diff_norm = diff_norm.sqrt();
            if diff_norm < tol {
                break;
            }
        }
        
        yout[k+1] = y.clone();
        f_vals[k+1] = f_next.clone();
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation AM5
pub fn solve_am5<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //uses AM4 to bootstrap 3 steps of history
    let mut y_boot: Vec<Vec<f64>> = vec![vec![0.0; n]; 5];
    let mut tgrid_boot = vec![0.0; 5];
    solve_am4(&f, t0, t0+4.0*h, y0, h, n, &mut y_boot, &mut tgrid_boot, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    for i in 0..5.min(y_boot.len()) {
        yout[i] = y_boot[i].clone();
    }
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    
    //computes initial F
    for i in 0..5 {
        f(tgrid[i], &yout[i], &mut f_vals[i]);
    }
    
    //defines the main AM5 solver
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_next = vec![0.0; n];
    for k in 4..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_next);
            for j in 0..n {
                y[j] = yout[k][j] + h * ((251.0/720.0) * f_next[j] + (646.0/720.0) * f_vals[k][j] -
                                          (264.0/720.0) * f_vals[k-1][j] + (106.0/720.0) * f_vals[k-2][j] -
                                          (19.0/720.0) * f_vals[k-3][j]);
            }
            let mut diff_norm = 0.0;
            for j in 0..n {
                let diff = y[j] - y_old[j];
                diff_norm += diff * diff;
            }
            diff_norm = diff_norm.sqrt();
            if diff_norm < tol {
                break;
            }
        }
        
        yout[k+1] = y.clone();
        f_vals[k+1] = f_next.clone();
    }
}
