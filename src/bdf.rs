//solves the nonlinear system of equations with a Gauss-Seidel relaxation BE (BDF1)
pub fn solve_be<F>(f: F, t0: f64, tf: f64, y0: &[f64],
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
    
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_val = vec![0.0; n];
    
    //defines the main BE solver
    for k in 0..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_val);
            for j in 0..n {
                y[j] = yout[k][j] + h * f_val[j];
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
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation BDF2
pub fn solve_bdf2<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                     h: f64, n: usize, yout: &mut [Vec<f64>],
                     tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //uses backward euler to bootstrap
    let mut y_be: Vec<Vec<f64>> = vec![vec![0.0; n]; 2];
    let mut tgrid_be = vec![0.0; 2];
    solve_be(&f, t0, t0+h, y0, h, n, &mut y_be, &mut tgrid_be, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    if y_be.len() > 1 {
        yout[1] = y_be[1].clone();
    }
    
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_val = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    
    for k in 1..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //defines the rhs
        for j in 0..n {
            rhs[j] = (-4.0 * yout[k][j] + yout[k-1][j]) / (2.0 * h);
        }
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_val);
            for j in 0..n {
                y[j] = (rhs[j] + f_val[j]) / (3.0 / (2.0 * h));
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
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation BDF3
pub fn solve_bdf3<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                     h: f64, n: usize, yout: &mut [Vec<f64>],
                     tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //bootstrap using BE then BDF2
    let mut y_be: Vec<Vec<f64>> = vec![vec![0.0; n]; 2];
    let mut tgrid_be = vec![0.0; 2];
    let mut y_bdf2: Vec<Vec<f64>> = vec![vec![0.0; n]; 3];
    let mut tgrid_bdf2 = vec![0.0; 3];
    solve_be(&f, t0, t0+h, y0, h, n, &mut y_be, &mut tgrid_be, sweeps, tol);
    solve_bdf2(&f, t0, t0+2.0*h, y0, h, n, &mut y_bdf2, &mut tgrid_bdf2, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    if y_be.len() > 1 {
        yout[1] = y_be[1].clone();
    }
    if y_bdf2.len() > 2 {
        yout[2] = y_bdf2[2].clone();
    }
    
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_val = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    
    for k in 2..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //defines the rhs
        for j in 0..n {
            rhs[j] = (-11.0 * yout[k][j] + 18.0 * yout[k-1][j] - 9.0 * yout[k-2][j] + 2.0 * yout[k-3][j]) / (6.0 * h);
        }
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_val);
            for j in 0..n {
                y[j] = (rhs[j] + f_val[j]) / (11.0 / (6.0 * h));
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
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation BDF4
pub fn solve_bdf4<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                     h: f64, n: usize, yout: &mut [Vec<f64>],
                     tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //bootstrap using BDF1–3
    let mut y_be: Vec<Vec<f64>> = vec![vec![0.0; n]; 2];
    let mut tgrid_be = vec![0.0; 2];
    let mut y_bdf2: Vec<Vec<f64>> = vec![vec![0.0; n]; 3];
    let mut tgrid_bdf2 = vec![0.0; 3];
    let mut y_bdf3: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];
    let mut tgrid_bdf3 = vec![0.0; 4];
    solve_be(&f, t0, t0+h, y0, h, n, &mut y_be, &mut tgrid_be, sweeps, tol);
    solve_bdf2(&f, t0, t0+2.0*h, y0, h, n, &mut y_bdf2, &mut tgrid_bdf2, sweeps, tol);
    solve_bdf3(&f, t0, t0+3.0*h, y0, h, n, &mut y_bdf3, &mut tgrid_bdf3, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    if y_be.len() > 1 {
        yout[1] = y_be[1].clone();
    }
    if y_bdf2.len() > 2 {
        yout[2] = y_bdf2[2].clone();
    }
    if y_bdf3.len() > 3 {
        yout[3] = y_bdf3[3].clone();
    }
    
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_val = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    
    for k in 3..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //defines the rhs
        for j in 0..n {
            rhs[j] = (-25.0 * yout[k][j] + 48.0 * yout[k-1][j] - 36.0 * yout[k-2][j] +
                      16.0 * yout[k-3][j] - 3.0 * yout[k-4][j]) / (12.0 * h);
        }
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_val);
            for j in 0..n {
                y[j] = (rhs[j] + f_val[j]) / (25.0 / (12.0 * h));
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
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation BDF5
pub fn solve_bdf5<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                     h: f64, n: usize, yout: &mut [Vec<f64>],
                     tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //bootstrap up to BDF4
    let mut y_be: Vec<Vec<f64>> = vec![vec![0.0; n]; 2];
    let mut tgrid_be = vec![0.0; 2];
    let mut y_bdf2: Vec<Vec<f64>> = vec![vec![0.0; n]; 3];
    let mut tgrid_bdf2 = vec![0.0; 3];
    let mut y_bdf3: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];
    let mut tgrid_bdf3 = vec![0.0; 4];
    let mut y_bdf4: Vec<Vec<f64>> = vec![vec![0.0; n]; 5];
    let mut tgrid_bdf4 = vec![0.0; 5];
    solve_be(&f, t0, t0+h, y0, h, n, &mut y_be, &mut tgrid_be, sweeps, tol);
    solve_bdf2(&f, t0, t0+2.0*h, y0, h, n, &mut y_bdf2, &mut tgrid_bdf2, sweeps, tol);
    solve_bdf3(&f, t0, t0+3.0*h, y0, h, n, &mut y_bdf3, &mut tgrid_bdf3, sweeps, tol);
    solve_bdf4(&f, t0, t0+4.0*h, y0, h, n, &mut y_bdf4, &mut tgrid_bdf4, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    if y_be.len() > 1 {
        yout[1] = y_be[1].clone();
    }
    if y_bdf2.len() > 2 {
        yout[2] = y_bdf2[2].clone();
    }
    if y_bdf3.len() > 3 {
        yout[3] = y_bdf3[3].clone();
    }
    if y_bdf4.len() > 4 {
        yout[4] = y_bdf4[4].clone();
    }
    
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_val = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    
    for k in 4..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //defines the rhs
        for j in 0..n {
            rhs[j] = (-137.0 * yout[k][j] + 300.0 * yout[k-1][j] - 300.0 * yout[k-2][j] +
                      200.0 * yout[k-3][j] - 75.0 * yout[k-4][j] + 12.0 * yout[k-5][j]) / (60.0 * h);
        }
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_val);
            for j in 0..n {
                y[j] = (rhs[j] + f_val[j]) / (137.0 / (60.0 * h));
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
    }
}

//solves the nonlinear system of equations with a Gauss-Seidel relaxation BDF6
pub fn solve_bdf6<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                     h: f64, n: usize, yout: &mut [Vec<f64>],
                     tgrid: &mut [f64], sweeps: usize, tol: f64)
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    //bootstrap up to BDF5
    let mut y_be: Vec<Vec<f64>> = vec![vec![0.0; n]; 2];
    let mut tgrid_be = vec![0.0; 2];
    let mut y_bdf2: Vec<Vec<f64>> = vec![vec![0.0; n]; 3];
    let mut tgrid_bdf2 = vec![0.0; 3];
    let mut y_bdf3: Vec<Vec<f64>> = vec![vec![0.0; n]; 4];
    let mut tgrid_bdf3 = vec![0.0; 4];
    let mut y_bdf4: Vec<Vec<f64>> = vec![vec![0.0; n]; 5];
    let mut tgrid_bdf4 = vec![0.0; 5];
    let mut y_bdf5: Vec<Vec<f64>> = vec![vec![0.0; n]; 6];
    let mut tgrid_bdf5 = vec![0.0; 6];
    solve_be(&f, t0, t0+h, y0, h, n, &mut y_be, &mut tgrid_be, sweeps, tol);
    solve_bdf2(&f, t0, t0+2.0*h, y0, h, n, &mut y_bdf2, &mut tgrid_bdf2, sweeps, tol);
    solve_bdf3(&f, t0, t0+3.0*h, y0, h, n, &mut y_bdf3, &mut tgrid_bdf3, sweeps, tol);
    solve_bdf4(&f, t0, t0+4.0*h, y0, h, n, &mut y_bdf4, &mut tgrid_bdf4, sweeps, tol);
    solve_bdf5(&f, t0, t0+5.0*h, y0, h, n, &mut y_bdf5, &mut tgrid_bdf5, sweeps, tol);
    
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    if y_be.len() > 1 {
        yout[1] = y_be[1].clone();
    }
    if y_bdf2.len() > 2 {
        yout[2] = y_bdf2[2].clone();
    }
    if y_bdf3.len() > 3 {
        yout[3] = y_bdf3[3].clone();
    }
    if y_bdf4.len() > 4 {
        yout[4] = y_bdf4[4].clone();
    }
    if y_bdf5.len() > 5 {
        yout[5] = y_bdf5[5].clone();
    }
    
    let mut y = vec![0.0; n];
    let mut y_old = vec![0.0; n];
    let mut f_val = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    
    for k in 5..nsteps {
        y.copy_from_slice(&yout[k]);
        
        //defines the rhs
        for j in 0..n {
            rhs[j] = (-147.0 * yout[k][j] + 360.0 * yout[k-1][j] - 450.0 * yout[k-2][j] +
                      400.0 * yout[k-3][j] - 225.0 * yout[k-4][j] + 72.0 * yout[k-5][j] -
                      10.0 * yout[k-6][j]) / (60.0 * h);
        }
        
        //implements Gauss-Seidel relaxation
        for _i in 0..sweeps {
            y_old.copy_from_slice(&y);
            f(tgrid[k+1], &y, &mut f_val);
            for j in 0..n {
                y[j] = (rhs[j] + f_val[j]) / (147.0 / (60.0 * h));
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
    }
}