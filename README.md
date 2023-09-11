# Filters
5 Files.

Each contain a self-made RK4 integrator.

projectile_Github_version.py : Solves projectile problem of linear dynamics and nonlinear measurements. Calculates true and nominal trajectory of projectile using self-made RK4. Resolves for initial conditions                                     using self-made Nonlinear Least Squares (NLS). Estimates trajectory given NLS outputs with self-made Extended Kalman Filter.

EKF_Github_version.py : Uses RK4 and self-made Extended Kalman Filter to solve a system of nonlinear dynamics and measurements.

UKF_Github_version.py : Uses RK4 and self-made Unscented Kalman Filter to solve a system of nonlinear dynamics and measurements.

EnKF_Github_version.py : Uses RK4 and self-made Ensemble Kalman Filter to solve a system of nonlinear dynamics and measurements.

BootstrapPF_Github_version.py : Uses RK4 and self-made Bootstrap Particle Filter to solve a system of linear dynamics and measurements.
                        Nonlinear dynamics and measurements cause severe particle degeneracy in this version.
                        Comes with stratified resampling and roughening.
