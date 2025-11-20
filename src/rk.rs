//defines the step for RK1
pub fn step_rk1<F>(f: F, t: f64, y: &[f64], h: f64, n: usize, y_next: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let mut k1 = vec![0.0; n];
    f(t, y, &mut k1);
    for i in 0..n {
        y_next[i] = y[i] + h * k1[i];
    }
}

//main solver for RK1
pub fn solve_rk1<F>(f: F, t0: f64, tf: f64, y0: &[f64],
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
    
    for k in 0..nsteps {
        let y_k = yout[k].clone();
        step_rk1(&f, tgrid[k], &y_k, h, n, &mut yout[k+1]);
    }
}

//defines the step for RK2
pub fn step_rk2<F>(f: F, t: f64, y: &[f64], h: f64, n: usize, y_next: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    f(t, y, &mut k1);
    let y_temp: Vec<f64> = (0..n).map(|i| y[i] + h * k1[i]).collect();
    f(t + h, &y_temp, &mut k2);
    for i in 0..n {
        y_next[i] = y[i] + (h / 2.0) * (k1[i] + k2[i]);
    }
}

//main solver for RK2
pub fn solve_rk2<F>(f: F, t0: f64, tf: f64, y0: &[f64],
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
    
    for k in 0..nsteps {
        let y_k = yout[k].clone();
        step_rk2(&f, tgrid[k], &y_k, h, n, &mut yout[k+1]);
    }
}

//defines the step for RK3
pub fn step_rk3<F>(f: F, t: f64, y: &[f64], h: f64, n: usize, y_next: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    f(t, y, &mut k1);
    let mut y_temp: Vec<f64> = (0..n).map(|i| y[i] + (h/2.0) * k1[i]).collect();
    f(t + h/2.0, &y_temp, &mut k2);
    y_temp = (0..n).map(|i| y[i] + h * (-k1[i] + 2.0*k2[i])).collect();
    f(t + h, &y_temp, &mut k3);
    for i in 0..n {
        y_next[i] = y[i] + (h / 6.0) * (k1[i] + 4.0*k2[i] + k3[i]);
    }
}

//main solver for RK3
pub fn solve_rk3<F>(f: F, t0: f64, tf: f64, y0: &[f64],
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
    
    for k in 0..nsteps {
        let y_k = yout[k].clone();
        step_rk3(&f, tgrid[k], &y_k, h, n, &mut yout[k+1]);
    }
}

//defines the step for RK4
pub fn step_rk4<F>(f: F, t: f64, y: &[f64], h: f64, n: usize, y_next: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    f(t, y, &mut k1);
    let mut y_temp: Vec<f64> = (0..n).map(|i| y[i] + (h/2.0) * k1[i]).collect();
    f(t + h/2.0, &y_temp, &mut k2);
    y_temp = (0..n).map(|i| y[i] + (h/2.0) * k2[i]).collect();
    f(t + h/2.0, &y_temp, &mut k3);
    y_temp = (0..n).map(|i| y[i] + h * k3[i]).collect();
    f(t + h, &y_temp, &mut k4);
    for i in 0..n {
        y_next[i] = y[i] + (h / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }
}

//main solver for RK4
pub fn solve_rk4<F>(f: F, t0: f64, tf: f64, y0: &[f64],
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
    
    for k in 0..nsteps {
        let y_k = yout[k].clone();
        step_rk4(&f, tgrid[k], &y_k, h, n, &mut yout[k+1]);
    }
}

//defines the step for RK5
pub fn step_rk5<F>(f: F, t: f64, y: &[f64], h: f64, n: usize, y_next: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    f(t, y, &mut k1);
    let mut y_temp: Vec<f64> = (0..n).map(|i| y[i] + (h/4.0) * k1[i]).collect();
    f(t + h/4.0, &y_temp, &mut k2);
    y_temp = (0..n).map(|i| y[i] + (h/8.0) * (k1[i] + k2[i])).collect();
    f(t + h/4.0, &y_temp, &mut k3);
    y_temp = (0..n).map(|i| y[i] + (h/2.0) * k3[i]).collect();
    f(t + h/2.0, &y_temp, &mut k4);
    y_temp = (0..n).map(|i| y[i] + (h/16.0) * (3.0*k1[i] + 9.0*k4[i])).collect();
    f(t + 3.0*h/4.0, &y_temp, &mut k5);
    y_temp = (0..n).map(|i| y[i] + (h/7.0) * (2.0*k1[i] + 3.0*k2[i] + 4.0*k4[i] - 12.0*k3[i])).collect();
    f(t + h, &y_temp, &mut k6);
    for i in 0..n {
        y_next[i] = y[i] + (h / 90.0) * (7.0*k1[i] + 32.0*k3[i] + 12.0*k4[i] + 32.0*k5[i] + 7.0*k6[i]);
    }
}

//main solver for RK5
pub fn solve_rk5<F>(f: F, t0: f64, tf: f64, y0: &[f64],
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
    
    for k in 0..nsteps {
        let y_k = yout[k].clone();
        step_rk5(&f, tgrid[k], &y_k, h, n, &mut yout[k+1]);
    }
}
