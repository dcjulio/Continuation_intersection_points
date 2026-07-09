
function H(x, alpha, beta, c, xi)
     y = copy(x);
     y[1] = alpha*x[1]*x[1] + beta*x[2] + c
     y[2] = x[1]
     y[3] = xi*x[3] + x[1]
     return y
end

#  Inverse of H
function Hinv(x, alpha, beta, c, xi)
     y = copy(x);
     y[1] = x[2]
     y[2] = (x[1] - c -alpha*x[2]*x[2])/beta;
     y[3] = (-x[2] + x[3])/xi;
     return y
end 

# Jacobian Matrix of H
function DH(x, alpha, beta, xi)
     J = [2*alpha*x[1] beta 0
         1 0 0
         1 0 xi]
     return J
end


# # =================== BOUNDARY VALUE PROBLEMS H2 ===================

function PlaneBVP(x, pars, xi)
    # returns y: the vector that has to be zeroed by the solver
    # y: [start at eigenvector, orbit_error, ends at plane]

    N  = div(length(x) - 1, 3)  
    y  = copy(x) # 3N+1
    hs = x[1] # scalar; initial distance along the stable manifold        
    alpha, beta, c, v = pars # v is the vector that defines the plane

    # orbit conditions
    orbit = reshape(x[2:end], (3, N)) # only the orbit, without the scalar hs: 3xN
    Hk(x) = Hinv(x, alpha, beta, c, xi) # because stable, then inverse map (from the point to the plane)
    H_orbit = mapslices(Hk, orbit[:, 1:N-1], dims=(1)) # mapping the points: 3×(N−1)
    orbit_error = H_orbit - orbit[:, 2:N] # 3×(N−1)   
    y[4:end-1] = orbit_error[:]' #start at eigenvector, orbit_error, ends at plane


    # === fixp
    rhoplus  = (-(beta-1) + sqrt((beta-1)^2 - 4*alpha*c))/(2*alpha)
    rhominus = (-(beta-1) - sqrt((beta-1)^2 - 4*alpha*c))/(2*alpha)
    pplus  = [rhoplus rhoplus rhoplus/(1-xi)]'
    pminus = [rhominus rhominus rhominus/(1-xi)]'

    # === Boundary conditions ===
    p = pminus;
    J = DH(p, alpha, beta, xi)                              
    E = eigen(J)
    s_idx = abs.(E.values) .< 1
    s_vec = E.vectors[:, s_idx]

    # we always want the stable vector to point in the positive x direction for our definition of "pos" and "neg" branches
    if s_vec[1] < 0 
        s_vec = -s_vec
    end

    x1 = orbit[:, 1] # starts at the manifold close to p
    xN = orbit[:, N] # last point of the orbit, should be on the plane

    y[1:3] = x1 - (p + hs*s_vec) # the first three elements of y are in a hs distance to p in the direction of the eigenvector
    y[end] = (v[2,2]-v[1,2])*(xN[1] - v[1,1]) - (v[2,1]-v[1,1])*(xN[2] - v[1,2]);   #plane condition

    #println("Initial point deviation from PM - hs * v_s: ", norm(y[1:3]))
    return y
end