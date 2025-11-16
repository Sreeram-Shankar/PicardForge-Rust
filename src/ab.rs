//defines the step for AB2
pub fn step_ab2(y_n: &[f64], f_n: &[f64], f_prev: &[f64], h: f64, n: usize, y_next: &mut [f64]) {
    for i in 0..n {
        y_next[i] = y_n[i] + h * ((3.0/2.0) * f_n[i] - (1.0/2.0) * f_prev[i]);
    }
}

//main solver for AB2
pub fn solve_ab2<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    
    f(tgrid[0], &yout[0], &mut f_vals[0]);
    
    //uses RK2 for first step
    f(tgrid[0], &yout[0], &mut k1);
    let mut y_temp: Vec<f64> = (0..n).map(|i| yout[0][i] + h * k1[i]).collect();
    f(tgrid[0] + h, &y_temp, &mut k2);
    yout[1] = (0..n).map(|i| yout[0][i] + 0.5 * h * (k1[i] + k2[i])).collect();
    f(tgrid[1], &yout[1], &mut f_vals[1]);
    
    for k in 1..nsteps {
        step_ab2(&yout[k], &f_vals[k], &f_vals[k-1], h, n, &mut yout[k+1]);
        f(tgrid[k+1], &yout[k+1], &mut f_vals[k+1]);
    }
}

//defines the step for AB3
pub fn step_ab3(y_n: &[f64], f_n: &[f64], f_prev: &[f64], f_prev2: &[f64],
                h: f64, n: usize, y_next: &mut [f64]) {
    for i in 0..n {
        y_next[i] = y_n[i] + h * ((23.0/12.0) * f_n[i] - (16.0/12.0) * f_prev[i] + (5.0/12.0) * f_prev2[i]);
    }
}

//main solver for AB3
pub fn solve_ab3<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    
    f(tgrid[0], &yout[0], &mut f_vals[0]);
    
    //uses RK3 for first two steps
    f(tgrid[0], &yout[0], &mut k1);
    let mut y_temp: Vec<f64> = (0..n).map(|i| yout[0][i] + (h/2.0) * k1[i]).collect();
    f(tgrid[0] + h/2.0, &y_temp, &mut k2);
    y_temp = (0..n).map(|i| yout[0][i] + h * (-k1[i] + 2.0*k2[i])).collect();
    f(tgrid[0] + h, &y_temp, &mut k3);
    yout[1] = (0..n).map(|i| yout[0][i] + (h/6.0) * (k1[i] + 4.0*k2[i] + k3[i])).collect();
    f(tgrid[1], &yout[1], &mut f_vals[1]);
    
    f(tgrid[1], &yout[1], &mut k1);
    y_temp = (0..n).map(|i| yout[1][i] + (h/2.0) * k1[i]).collect();
    f(tgrid[1] + h/2.0, &y_temp, &mut k2);
    y_temp = (0..n).map(|i| yout[1][i] + h * (-k1[i] + 2.0*k2[i])).collect();
    f(tgrid[1] + h, &y_temp, &mut k3);
    yout[2] = (0..n).map(|i| yout[1][i] + (h/6.0) * (k1[i] + 4.0*k2[i] + k3[i])).collect();
    f(tgrid[2], &yout[2], &mut f_vals[2]);
    
    for k in 2..nsteps {
        step_ab3(&yout[k], &f_vals[k], &f_vals[k-1], &f_vals[k-2], h, n, &mut yout[k+1]);
        f(tgrid[k+1], &yout[k+1], &mut f_vals[k+1]);
    }
}

//defines the step for AB4
pub fn step_ab4(y_n: &[f64], f_n: &[f64], f_prev: &[f64], f_prev2: &[f64], f_prev3: &[f64],
                h: f64, n: usize, y_next: &mut [f64]) {
    for i in 0..n {
        y_next[i] = y_n[i] + h * ((55.0/24.0) * f_n[i] - (59.0/24.0) * f_prev[i] +
                                    (37.0/24.0) * f_prev2[i] - (9.0/24.0) * f_prev3[i]);
    }
}

//main solver for AB4
pub fn solve_ab4<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    
    f(tgrid[0], &yout[0], &mut f_vals[0]);
    
    //uses RK4 for first three steps
    for i in 0..3 {
        f(tgrid[i], &yout[i], &mut k1);
        let mut y_temp: Vec<f64> = (0..n).map(|j| yout[i][j] + (h/2.0) * k1[j]).collect();
        f(tgrid[i] + h/2.0, &y_temp, &mut k2);
        y_temp = (0..n).map(|j| yout[i][j] + (h/2.0) * k2[j]).collect();
        f(tgrid[i] + h/2.0, &y_temp, &mut k3);
        y_temp = (0..n).map(|j| yout[i][j] + h * k3[j]).collect();
        f(tgrid[i] + h, &y_temp, &mut k4);
        yout[i+1] = (0..n).map(|j| yout[i][j] + (h/6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])).collect();
        f(tgrid[i+1], &yout[i+1], &mut f_vals[i+1]);
    }
    
    for k in 3..nsteps {
        step_ab4(&yout[k], &f_vals[k], &f_vals[k-1], &f_vals[k-2], &f_vals[k-3], h, n, &mut yout[k+1]);
        f(tgrid[k+1], &yout[k+1], &mut f_vals[k+1]);
    }
}

//defines the step for AB5
pub fn step_ab5(y_n: &[f64], f_n: &[f64], f_prev: &[f64], f_prev2: &[f64],
                f_prev3: &[f64], f_prev4: &[f64], h: f64, n: usize, y_next: &mut [f64]) {
    for i in 0..n {
        y_next[i] = y_n[i] + h * ((1901.0/720.0) * f_n[i] - (2774.0/720.0) * f_prev[i] +
                                    (2616.0/720.0) * f_prev2[i] - (1274.0/720.0) * f_prev3[i] +
                                    (251.0/720.0) * f_prev4[i]);
    }
}

//main solver for AB5
pub fn solve_ab5<F>(f: F, t0: f64, tf: f64, y0: &[f64],
                    h: f64, n: usize, yout: &mut [Vec<f64>],
                    tgrid: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let nsteps = ((tf - t0) / h).ceil() as usize;
    
    for k in 0..=nsteps {
        tgrid[k] = t0 + k as f64 * h;
    }
    
    yout[0] = y0.to_vec();
    
    let mut f_vals: Vec<Vec<f64>> = vec![vec![0.0; n]; nsteps + 1];
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    
    f(tgrid[0], &yout[0], &mut f_vals[0]);
    
    //uses RK4 for first four steps
    for i in 0..4 {
        f(tgrid[i], &yout[i], &mut k1);
        let mut y_temp: Vec<f64> = (0..n).map(|j| yout[i][j] + (h/2.0) * k1[j]).collect();
        f(tgrid[i] + h/2.0, &y_temp, &mut k2);
        y_temp = (0..n).map(|j| yout[i][j] + (h/2.0) * k2[j]).collect();
        f(tgrid[i] + h/2.0, &y_temp, &mut k3);
        y_temp = (0..n).map(|j| yout[i][j] + h * k3[j]).collect();
        f(tgrid[i] + h, &y_temp, &mut k4);
        yout[i+1] = (0..n).map(|j| yout[i][j] + (h/6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])).collect();
        f(tgrid[i+1], &yout[i+1], &mut f_vals[i+1]);
    }
    
    for k in 4..nsteps {
        step_ab5(&yout[k], &f_vals[k], &f_vals[k-1], &f_vals[k-2], &f_vals[k-3], &f_vals[k-4], h, n, &mut yout[k+1]);
        f(tgrid[k+1], &yout[k+1], &mut f_vals[k+1]);
    }
}
