using ProximalAlgorithms
using ProximalOperators
using LinearAlgebra
using Random
using Test

@testset "Nonconvex QP (tiny, $T)" for T in [Float64]
    Q = Matrix(Diagonal(T[-0.5, 1.0]))
    q = T[0.3, 0.5]
    low = T(-1.0)
    upp = T(+1.0)

    f = Quadratic(Q, q)
    g = IndBox(low, upp)

    n = 2

    Lip = maximum(diag(Q))
    gamma = T(0.95) / Lip

    @testset "PANOC" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOC()
        x, it = solver(x0, f = f, g = g)
        z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
        @test norm(x - z, Inf) / gamma <= solver.tol
        @test x0 == x0_backup
    end

    @testset "PANOCplus" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.PANOCplus()
        x, it = solver(x0, f = f, g = g)
        z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
        @test norm(x - z, Inf) / gamma <= solver.tol
        @test x0 == x0_backup
    end

    @testset "ZeroFPR" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.ZeroFPR()
        x, it = solver(x0, f = f, g = g)
        z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
        @test norm(x - z, Inf) / gamma <= solver.tol
        @test x0 == x0_backup
    end

    @testset "LiLin" begin
        x0 = zeros(T, n)
        x0_backup = copy(x0)
        solver = ProximalAlgorithms.LiLin(gamma = gamma)
        x, it = solver(x0, f = f, g = g)
        z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
        @test norm(x - z, Inf) / gamma <= solver.tol
        @test x0 == x0_backup
    end
end

@testset "Nonconvex QP (small, $T)" for T in [Float64]
    @testset "Random problem $k" for k = 1:5
        Random.seed!(k)

        n = 100
        A = randn(T, n, n)
        U, R = qr(A)
        eigenvalues = T(2) .* rand(T, n) .- T(1)
        Q = U * Diagonal(eigenvalues) * U'
        Q = 0.5 * (Q + Q')
        q = randn(T, n)

        low = T(-1.0)
        upp = T(+1.0)

        f = Quadratic(Q, q)
        g = IndBox(low, upp)

        Lip = maximum(abs.(eigenvalues))
        gamma = T(0.95) / Lip

        TOL = 1e-4

        @testset "PANOC" begin
            x0 = zeros(T, n)
            x0_backup = copy(x0)
            solver = ProximalAlgorithms.PANOC(tol = TOL)
            x, it = solver(x0, f = f, g = g)
            z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
            @test norm(x - z, Inf) / gamma <= solver.tol
            @test x0 == x0_backup
        end

        @testset "PANOCplus" begin
            x0 = zeros(T, n)
            x0_backup = copy(x0)
            solver = ProximalAlgorithms.PANOCplus(tol = TOL)
            x, it = solver(x0, f = f, g = g)
            z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
            @test norm(x - z, Inf) / gamma <= solver.tol
            @test x0 == x0_backup
        end

        @testset "ZeroFPR" begin
            x0 = zeros(T, n)
            x0_backup = copy(x0)
            solver = ProximalAlgorithms.ZeroFPR(tol = TOL)
            x, it = solver(x0, f = f, g = g)
            z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
            @test norm(x - z, Inf) / gamma <= solver.tol
            @test x0 == x0_backup
        end

        @testset "LiLin" begin
            x0 = zeros(T, n)
            x0_backup = copy(x0)
            solver = ProximalAlgorithms.LiLin(gamma = gamma, tol = TOL)
            x, it = solver(x0, f = f, g = g)
            z = min.(upp, max.(low, x .- gamma .* (Q * x + q)))
            @test norm(x - z, Inf) / gamma <= solver.tol
            @test x0 == x0_backup
        end
    end
end
