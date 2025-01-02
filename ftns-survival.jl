# load pacakges
using Pkg, Plots, Plots.PlotMeasures, ColorSchemes, StatsPlots
using DataFrames, CSV, LinearAlgebra, StatsBase, Distributions, Random, SpecialFunctions, LaTeXStrings, JLD2, BenchmarkTools, Kronecker, Interpolations

using Suppressor: @suppress_err
@suppress_err using Interpolations

function blockstr(p, diag, offdiag, bs)
    x, y = zeros(p, p), zeros(bs, bs)

    for i in 1 : bs
        for j in (i + 1) : bs
            y[i, j] = y[j, i] = offdiag ^ abs(i - j)# 0.9 - 0.05 * abs(i - j)
        end
    end

    for i = 1 : Int(p / bs); x[(bs * (i - 1) + 1): (bs * i), (bs * (i - 1) + 1): (bs * i)] = y; end
    for i = 1 : p; x[i, i] = diag; end
    return x
end

function fcovstr(p, offdiag)
    x = zeros(p, p)

    for i in 1 : p
        for j in i : p
            x[i, j] = x[j, i] = offdiag ^ abs(i - j)# 0.9 - 0.05 * abs(i - j)
        end
    end
    return x
end

## find roots of ax^2 + bx + c
function quadratic(a, b, c)
    discr = b^2 - 4*a*c
    sq = (discr > 0) ? sqrt(discr) : sqrt(discr + 0im)
    x1, x2 = (-b - sq) / (2a), (-b + sq) / (2a)
    if x1 > x2; x2, x1; else  x1, x2; end
end

function detR(R, r, i, j)
    R0 = copy(R)
    R0[i, j] = R0[j, i] = r
    detRr = det(R0)
    detRr
end

function llik_r(rr, ri, rj, wi, S2g, R, zmuz, q, n) 
    R0 = copy(R)
    R0[ri, rj] = R0[rj, ri] = rr
    
    S = S2g .* R0
    S = 0.5 * (S + S')
    Schol = cholesky(S)
    
    sum_zmuz = 0.0; for i = 1 : n; sum_zmuz += wi[i] * dot(zmuz[i, :], Schol \ zmuz[i, :]); end
    -0.5 * sum(wi) * logdet(Schol) - 0.5 * sum_zmuz
end

mutable struct Hyperparameters
    n::Int64 # sample size
    p::Int64 # dimension of X  
    q::Int64 # dimension of Z

    m_w::Float64
    s2_w::Float64
    
    s2_logs::Float64 #  logs ~ N(0, sqrt(s2_logs))
    
    d_wi::Float64
    d_s2log::Float64
end

function construct_H(X, Z)
    n = size(X)[1] # sample size
    p = size(X)[2] # dimension of X  
    q = size(Z)[2] # dimension of Z

    m_w, s2_w = 5.0, 3.0
    s2_logs = 100 # logs ~ N(0, sqrt(s2logs))
    
    # step size in MH updates
    d_wi = 5.0
    d_s2log = 0.2
    
    return Hyperparameters(n, p, q, m_w, s2_w, s2_logs, d_wi, d_s2log)
end

struct Result
    b::Array{Float64, 2}
    gs::Array{Float64, 3}
    nu::Array{Float64, 2}
    lambda::Array{Float64, 2}
    tau2b::Array{Float64, 1}
    tau2g::Array{Float64, 2}
    xib::Array{Float64, 1}
    xig::Array{Float64, 2}
    u::Array{Float64, 2}
    su::Array{Float64, 1}
    sg::Array{Float64, 2}
    wi::Array{Float64, 2}
    R::Array{Float64, 3}
    elapsed_time::Float64
end

function f_add_diagonal(x, tau, c)
    xx = copy(x)
    @inbounds for s = 1 : size(x)[1]; xx[s, s] += tau * c[s]; end
    xx
end

function f_sum_x2(x, y)
    sum_x2 = 0.0
    @inbounds for s = 1 : length(x); sum_x2 += x[s] * x[s] * y[s]; end
    sum_x2
end

function update_g!(gs, wi, X, Z, S_, tau_, lambda_, p, q, n)
    
    xxS_z  = zeros(p, q)
    wxxt = zeros(p, p)
    for i in 1 : n
        x = X[i, :]
        xt = transpose(x)
        wx = wi[i] * x
        wxxt += wx * xt
        for ll in 1 : q; xxS_z[:, ll] += dot(S_[:, ll], Z[i, :]) * wx; end
    end
    
    for ll in 1 : q; 

        xxg_ll = zeros(p, 1); for j in 1 : q; if j != ll; xxg_ll += S_[ll, j] * wxxt * gs[:, j]; end; end

        m0 = xxS_z[:, ll] - xxg_ll
        M_ = S_[ll, ll] * wxxt
        M_ = f_add_diagonal(M_, tau_[ll], lambda_)
        M_ = (M_ + M_') * 0.5    
        M_chol = cholesky(M_)
        m = M_chol \ m0

        rn = randn(p)
        gs[:, ll] = m + M_chol.U \ rn
    end
end
function update_sg!(sg, sg_, wi, R_, xig, zmuz, q, n, s2_logs, d_s2log)
    
    wzmuz = zeros(q, q); for i in 1 : n; ei = zmuz[i, :]; eit = transpose(ei); wzmuz += wi[i] * ei * eit; end   
    for ll in 1 : q; 
        
        c1, c2 = 0.0, 0.0
        for hh = 1 : q; if hh != ll; c1 += sg_[hh] * R_[ll, hh] * wzmuz[ll, hh]; end; end
        c2 = wzmuz[ll, ll]
        
        curr = sg[ll] 
        logcurr = log(curr) 

        lognew = rand(Normal(logcurr, d_s2log)) 
        new = exp(lognew) 

        c1cur = 2 * c1 * sg_[ll] 
        c2cur = (R_[ll, ll] * c2 + 2 / xig[ll]) / (curr * curr)

        c1new = 2 * c1 / new 
        c2new = (R_[ll, ll] * c2 + 2 / xig[ll]) / (new * new)
        
        sum_wi = sum(wi)
        logr_cur = -(sum_wi + 1) * logcurr - 0.5 * (c1cur + c2cur + logcurr^2 / s2_logs)
        logr_new = -(sum_wi + 1) * lognew - 0.5 * (c1new + c2new + lognew^2 / s2_logs)

        logr = logr_new - logr_cur
        if log(rand()) < logr; sg[ll] = new; sg_[ll] = 1 / sg[ll]; end
        
    end
end

function update_su!(su, su_, u, Xb, xib, n, s2_logs, d_s2log)

    u_xb = u - Xb
    sum_uxb2 = dot(u_xb, u_xb)
    
    curr = su
    logcurr = log(su)
    
    lognew = rand(Normal(logcurr, d_s2log))
    new = exp(lognew)
    
    c1 = sum_uxb2 + 2 / xib
    logr_cur = -(n + 1) * logcurr - 0.5 * (c1 / (curr * curr) + logcurr ^ 2 / s2_logs)
    logr_new = -(n + 1) * lognew - 0.5 * (c1 / (new * new) + lognew ^ 2 / s2_logs)
    
    logr = logr_new - logr_cur
    if log(rand()) < logr; su = new; su_ = 1 / su; end
    
    return su, su_
end

logit(x) = log(x) - log(1 - x)
expit(x) = 1 / (1 + exp(-x))

function run_sampler(y, X, Z, H; d = zeros(100), n_total = 3000, use_w = true)
    
    n, p, q = H.n, H.p, H.q
    
    Xt = convert(Matrix, X')
    XtX = Xt * X
    XtX = (XtX + XtX') * 0.5

    ##### Initial values
    su, su_ = 1.0, 1.0
    s2u = su * su
    s2u_ = su_ * su_
    
    sg = 1.0 * ones(q)
    sg_ = 1 ./ sg

    R = blockstr(q, 1.0, 0.1, q)

    nu = [rand(InverseGamma(0.5, 1)) for i = 1 : p]
    lambda = [rand(InverseGamma(0.5, 1 / nu[i])) for i = 1 : p]
    lambda_ = 1 ./ lambda
    xib = rand(InverseGamma(0.5, su ^ 2)) 
    tau2b = rand(InverseGamma(0.5, 1 / xib))
    tau2b_ = 1 / tau2b
    xig = [rand(InverseGamma(0.5, sg_[i] ^ 2)) for i = 1 : q]
    tau2g = [rand(InverseGamma(0.5, 1 / xig[i])) for i = 1 : q] 
    tau2g_ = 1 ./ tau2g

    if use_w == true; wi = [0.99 for i = 1 : n]; else wi = ones(n); end

    R_ = inv(R)
    R_ = 0.5 * (R_ + R_')

    S2g = sg * sg'
    S_ = sg_ * sg_' .* R_
    
    u = log.(y)
    logy = log.(y)
    zmuz = zeros(n, q)
    b, b2 = zeros(p), zeros(p)
    gs = 1e-5 * ones(p, q) 

    b_r = zeros(n_total, p)
    gs_r = zeros(n_total, p, q)
    lambda_r = zeros(n_total, p)
    nu_r = zeros(n_total, p)
    tau2b_r = zeros(n_total)
    tau2g_r = zeros(n_total, q)
    xib_r = zeros(n_total)
    xig_r = zeros(n_total, q)
    u_r = zeros(n_total, n)
    w_r = zeros(n_total, n)
    R_r = zeros(n_total, q, q)
    sg_r = zeros(n_total, q)
    su_r = zeros(n_total)

    println("n = $n, n_total = $n_total")
    print("Running... ")    
    
    elapsed_time = (
    @suppress_err begin
    @elapsed for iter = 1 : n_total
                
                # update b
                Xu = Xt * u
                Sb_ = f_add_diagonal(s2u_ * XtX, tau2b_, lambda_)
                Sb_ = (Sb_ + Sb_') * 0.5 
                Sb_chol = cholesky(Sb_)

                mu_b = Sb_chol \ (s2u_ * Xu)
                rn = randn(p)
                b = mu_b + Sb_chol.U \ rn

                # update g
                update_g!(gs, wi, X, Z, S_, tau2g_, lambda_, p, q, n)
                mu_z = [[dot(X[j, :], gs[:, i]) for i = 1 : q] for j = 1 : n];
                for i = 1 : n; zmuz[i, :] = Z[i, :] - mu_z[i]; end

                # update lambda
                for j = 1 : p
                    b2[j] = b[j] * b[j]
                    sum2gj = 0.0; sum2gj = f_sum_x2(gs[j, :], tau2g_)
                    b_lambda = 0.5 * tau2b_ * b2[j] + 0.5 * sum2gj + 1 / nu[j]
                    lambda[j] = rand(InverseGamma(0.5 * q + 1.0, b_lambda))
                    lambda_[j] = 1 / lambda[j]
                end

                # update nu
                for j = 1 : p; nu[j] = rand(InverseGamma(1.0, lambda_[j] + 1.0)); end

                # update tau2
                tau2b = rand(InverseGamma(0.5 * p + 0.5, 0.5 * dot(b2, lambda_) + 1 / xib))
                tau2b_ = 1 / tau2b
                for l = 1 : q; tau2g[l] = rand(InverseGamma(0.5 * p + 0.5, 0.5 * f_sum_x2(gs[:, l], lambda_) + 1 / xig[l])); tau2g_[l] = 1 / tau2g[l]; end

                # update xi
                xib = rand(InverseGamma(1.0, tau2b_ + s2u_))
                for l = 1 : q; xig[l] = rand(InverseGamma(1.0, tau2g_[l] + sg_[l] * sg_[l])); end; 

                # update u
                Xb = X * b
                for i = 1 : n; if d[i] == 0; u[i] = rand(Truncated(Normal(Xb[i], su), logy[i], Inf)); end; end

                # update R
                for rj in 1 : (q - 1); 
                    for ri in (rj + 1) : q;
                        f1 = detR(R, 1, ri, rj)
                        f0 = detR(R, 0, ri, rj)
                        f_1 = detR(R, -1, ri, rj)
                        aa = f1 + f_1 - 2 * f0
                        bb = 0.5 * (f1 - f_1)
                        cc = f0

                        sol = quadratic(aa, bb, cc)
                        if aa < 0; r_grid = sol[1] : 0.02 : sol[2]; else r_grid = union(-1:0.02:sol[1], sol[2]:0.02:1); end # Haven't seen aa > 0 yet. 

                        nr_grid = length(r_grid)
                        logww = zeros(nr_grid)
                        ww = zeros(nr_grid)
                        for l in 1 : nr_grid
                            logww[l] = llik_r(r_grid[l], ri, rj, wi, S2g, R, zmuz, q, n)
                        end
                        logww = logww .- maximum(logww)
                        ww = exp.(logww)
                        ww = ww ./ sum(ww)

                        # To resolve an issue with interpolation
                        r_grid = [sol[1] - 1e-10; r_grid; sol[2] + 1e-10]; 
                        nr_grid = length(r_grid)
                        ww = [1e-10; ww; 1e-10]
                        ww = ww ./ sum(ww)

                        # step2: Approximate the inverse CDF
                        w_cdf = cumsum(ww) 
                        itp_linear = LinearInterpolation(w_cdf, r_grid)

                        # step3: Generate a uniform on [0, 1] and invert the approximate CDF
                        rn = rand()
                        rr = itp_linear.itp[rn]
                        R[ri, rj] = R[rj, ri] = rr;
                    end;
                end

                S = S2g .* R
                S = 0.5 * (S + S')
                Schol = cholesky(S)
                logdetS = logdet(Schol)
                
                # update w
                for i in 1 : n
                    logit_wcur = logit(wi[i])
                    logit_wstar = rand(Normal(logit_wcur, H.d_wi))
                    wstar = expit(logit_wstar)
                    zz2 = dot(zmuz[i, :], Schol \ zmuz[i, :])
                    logr_cur = -0.5 * q * wi[i] * log(2 * pi) - 0.5 * wi[i] * logdetS - 0.5 * wi[i] * zz2 - 0.5 * (logit_wcur - H.m_w) ^ 2 / H.s2_w
                    logr_prop = -0.5 * q * wstar * log(2 * pi) - 0.5 * wstar * logdetS - 0.5 * wstar * zz2 - 0.5 * (logit_wstar - H.m_w) ^ 2 / H.s2_w
                    logr = logr_prop - logr_cur 
                    if log(rand()) < logr; wi[i] = wstar; end
                end

                # update su
                su, su_ = update_su!(su, su_, u, Xb, xib, n, H.s2_logs, H.d_s2log) 
                s2u_ = su_ * su_

                # update sg
                L_ = inv(Schol.L) * diagm(sg)
                R_ = L_' * L_

                update_sg!(sg, sg_, wi, R_, xig, zmuz, q, n, H.s2_logs, H.d_s2log)
                S2g = sg * sg'
                S_ = sg_ * sg_' .* R_
                
                # save
                b_r[iter, :] = b
                gs_r[iter, :, :] = gs
                nu_r[iter, :] = nu
                lambda_r[iter, :] = lambda
                tau2b_r[iter] = tau2b
                tau2g_r[iter, :] = tau2g
                xib_r[iter] = xib
                xig_r[iter, :] = xig
                u_r[iter, :] = u
                su_r[iter] = su
                sg_r[iter, :] = sg
                w_r[iter, :] = wi
                R_r[iter, :, :] = R
            end
        end
    )
    println("complete.")
    println("Elapsed time = $elapsed_time seconds")
    
    return Result(b_r, gs_r, nu_r, lambda_r, tau2b_r, tau2g_r, xib_r, xig_r, u_r, su_r, sg_r, w_r, R_r, elapsed_time)
end