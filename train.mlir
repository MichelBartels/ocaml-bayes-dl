module @jit_optim attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<784x512xf32>, %arg1: tensor<784x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512x16xf32>, %arg5: tensor<512x16xf32>, %arg6: tensor<16xf32>, %arg7: tensor<16xf32>, %arg8: tensor<512x16xf32>, %arg9: tensor<512x16xf32>, %arg10: tensor<16xf32>, %arg11: tensor<16xf32>, %arg12: tensor<16x512xf32>, %arg13: tensor<16x512xf32>, %arg14: tensor<512xf32>, %arg15: tensor<512xf32>, %arg16: tensor<512x784xf32>, %arg17: tensor<512x784xf32>, %arg18: tensor<784xf32>, %arg19: tensor<784xf32>, %arg20: tensor<i32>, %arg21: tensor<784x512xf32>, %arg22: tensor<784x512xf32>, %arg23: tensor<512xf32>, %arg24: tensor<512xf32>, %arg25: tensor<512x16xf32>, %arg26: tensor<512x16xf32>, %arg27: tensor<16xf32>, %arg28: tensor<16xf32>, %arg29: tensor<512x16xf32>, %arg30: tensor<512x16xf32>, %arg31: tensor<16xf32>, %arg32: tensor<16xf32>, %arg33: tensor<16x512xf32>, %arg34: tensor<16x512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512x784xf32>, %arg38: tensor<512x784xf32>, %arg39: tensor<784xf32>, %arg40: tensor<784xf32>, %arg41: tensor<784x512xf32>, %arg42: tensor<784x512xf32>, %arg43: tensor<512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<512x16xf32>, %arg46: tensor<512x16xf32>, %arg47: tensor<16xf32>, %arg48: tensor<16xf32>, %arg49: tensor<512x16xf32>, %arg50: tensor<512x16xf32>, %arg51: tensor<16xf32>, %arg52: tensor<16xf32>, %arg53: tensor<16x512xf32>, %arg54: tensor<16x512xf32>, %arg55: tensor<512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512x784xf32>, %arg58: tensor<512x784xf32>, %arg59: tensor<784xf32>, %arg60: tensor<784xf32>, %arg61: tensor<128x784xf32>) -> (tensor<784x512xf32> {jax.result_info = "[0][0][0]"}, tensor<784x512xf32> {jax.result_info = "[0][0][1]"}, tensor<512xf32> {jax.result_info = "[0][1][0]"}, tensor<512xf32> {jax.result_info = "[0][1][1]"}, tensor<512x16xf32> {jax.result_info = "[0][2][0]"}, tensor<512x16xf32> {jax.result_info = "[0][2][1]"}, tensor<16xf32> {jax.result_info = "[0][3][0]"}, tensor<16xf32> {jax.result_info = "[0][3][1]"}, tensor<512x16xf32> {jax.result_info = "[0][4][0]"}, tensor<512x16xf32> {jax.result_info = "[0][4][1]"}, tensor<16xf32> {jax.result_info = "[0][5][0]"}, tensor<16xf32> {jax.result_info = "[0][5][1]"}, tensor<16x512xf32> {jax.result_info = "[0][6][0]"}, tensor<16x512xf32> {jax.result_info = "[0][6][1]"}, tensor<512xf32> {jax.result_info = "[0][7][0]"}, tensor<512xf32> {jax.result_info = "[0][7][1]"}, tensor<512x784xf32> {jax.result_info = "[0][8][0]"}, tensor<512x784xf32> {jax.result_info = "[0][8][1]"}, tensor<784xf32> {jax.result_info = "[0][9][0]"}, tensor<784xf32> {jax.result_info = "[0][9][1]"}, tensor<i32> {jax.result_info = "[0][10][0].count"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].mu[0][0]"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].mu[0][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[1][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[1][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[2][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[2][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[3][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[3][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[4][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[4][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[5][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[5][1]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].mu[6][0]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].mu[6][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[7][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[7][1]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].mu[8][0]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].mu[8][1]"}, tensor<784xf32> {jax.result_info = "[0][10][0].mu[9][0]"}, tensor<784xf32> {jax.result_info = "[0][10][0].mu[9][1]"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].nu[0][0]"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].nu[0][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[1][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[1][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[2][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[2][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[3][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[3][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[4][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[4][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[5][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[5][1]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].nu[6][0]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].nu[6][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[7][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[7][1]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].nu[8][0]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].nu[8][1]"}, tensor<784xf32> {jax.result_info = "[0][10][0].nu[9][0]"}, tensor<784xf32> {jax.result_info = "[0][10][0].nu[9][1]"}, tensor<f32> {jax.result_info = "[1]"}) {
    %cst = stablehlo.constant dense<-9.99999974E-5> : tensor<f32>
    %cst_0 = stablehlo.constant dense<9.99999974E-5> : tensor<f32>
    %cst_1 = stablehlo.constant dense<9.99999993E-9> : tensor<f32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<2147483647> : tensor<i32>
    %cst_3 = stablehlo.constant dense<9.990000e-01> : tensor<f32>
    %cst_4 = stablehlo.constant dense<1.000000e-03> : tensor<f32>
    %cst_5 = stablehlo.constant dense<0.899999976> : tensor<f32>
    %cst_6 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %cst_7 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %cst_8 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %cst_9 = stablehlo.constant dense<7.840000e+02> : tensor<f32>
    %cst_10 = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %cst_11 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %cst_12 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_14 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<128x784x512xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<128x512xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<128x512x16xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<128x16xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<128x1x16xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<128x16x512xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<128x512x784xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<128x784xf32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x784x512xf32>) -> tensor<128x784x512xf32>
    %2 = stablehlo.multiply %1, %cst_16 : tensor<128x784x512xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x784x512xf32>) -> tensor<128x784x512xf32>
    %5 = stablehlo.add %4, %2 : tensor<128x784x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %8 = stablehlo.multiply %7, %cst_17 : tensor<128x512xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %11 = stablehlo.add %10, %8 : tensor<128x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg61, dims = [0, 2] : (tensor<128x784xf32>) -> tensor<128x1x784xf32>
    %13 = stablehlo.dot_general %12, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x784xf32>, tensor<128x784x512xf32>) -> tensor<128x1x512xf32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x1x512xf32>
    %16 = stablehlo.tanh %15 : tensor<128x1x512xf32>
    %17 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1x512xf32>
    %18 = stablehlo.subtract %17, %16 : tensor<128x1x512xf32>
    %19 = stablehlo.broadcast_in_dim %arg5, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %21 = stablehlo.multiply %20, %cst_18 : tensor<128x512x16xf32>
    %22 = stablehlo.broadcast_in_dim %arg4, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %24 = stablehlo.add %23, %21 : tensor<128x512x16xf32>
    %25 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %27 = stablehlo.multiply %26, %cst_19 : tensor<128x16xf32>
    %28 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %30 = stablehlo.add %29, %27 : tensor<128x16xf32>
    %31 = stablehlo.dot_general %16, %24, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x512xf32>, tensor<128x512x16xf32>) -> tensor<128x1x16xf32>
    %32 = stablehlo.broadcast_in_dim %30, dims = [0, 2] : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
    %33 = stablehlo.add %31, %32 : tensor<128x1x16xf32>
    %34 = stablehlo.broadcast_in_dim %arg9, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %36 = stablehlo.multiply %35, %cst_18 : tensor<128x512x16xf32>
    %37 = stablehlo.broadcast_in_dim %arg8, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %39 = stablehlo.add %38, %36 : tensor<128x512x16xf32>
    %40 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %42 = stablehlo.multiply %41, %cst_19 : tensor<128x16xf32>
    %43 = stablehlo.broadcast_in_dim %arg10, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %45 = stablehlo.add %44, %42 : tensor<128x16xf32>
    %46 = stablehlo.dot_general %16, %39, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x512xf32>, tensor<128x512x16xf32>) -> tensor<128x1x16xf32>
    %47 = stablehlo.broadcast_in_dim %45, dims = [0, 2] : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
    %48 = stablehlo.add %46, %47 : tensor<128x1x16xf32>
    %49 = stablehlo.exponential %48 : tensor<128x1x16xf32>
    %50 = stablehlo.multiply %49, %cst_20 : tensor<128x1x16xf32>
    %51 = stablehlo.add %33, %50 : tensor<128x1x16xf32>
    %52 = stablehlo.broadcast_in_dim %arg13, dims = [1, 2] : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x16x512xf32>) -> tensor<128x16x512xf32>
    %54 = stablehlo.multiply %53, %cst_21 : tensor<128x16x512xf32>
    %55 = stablehlo.broadcast_in_dim %arg12, dims = [1, 2] : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0, 1, 2] : (tensor<1x16x512xf32>) -> tensor<128x16x512xf32>
    %57 = stablehlo.add %56, %54 : tensor<128x16x512xf32>
    %58 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %60 = stablehlo.multiply %59, %cst_17 : tensor<128x512xf32>
    %61 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %63 = stablehlo.add %62, %60 : tensor<128x512xf32>
    %64 = stablehlo.dot_general %51, %57, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x16xf32>, tensor<128x16x512xf32>) -> tensor<128x1x512xf32>
    %65 = stablehlo.broadcast_in_dim %63, dims = [0, 2] : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %66 = stablehlo.add %64, %65 : tensor<128x1x512xf32>
    %67 = stablehlo.tanh %66 : tensor<128x1x512xf32>
    %68 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1x512xf32>
    %69 = stablehlo.subtract %68, %67 : tensor<128x1x512xf32>
    %70 = stablehlo.broadcast_in_dim %arg17, dims = [1, 2] : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [0, 1, 2] : (tensor<1x512x784xf32>) -> tensor<128x512x784xf32>
    %72 = stablehlo.multiply %71, %cst_22 : tensor<128x512x784xf32>
    %73 = stablehlo.broadcast_in_dim %arg16, dims = [1, 2] : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1, 2] : (tensor<1x512x784xf32>) -> tensor<128x512x784xf32>
    %75 = stablehlo.add %74, %72 : tensor<128x512x784xf32>
    %76 = stablehlo.broadcast_in_dim %arg19, dims = [1] : (tensor<784xf32>) -> tensor<1x784xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<128x784xf32>
    %78 = stablehlo.multiply %77, %cst_23 : tensor<128x784xf32>
    %79 = stablehlo.broadcast_in_dim %arg18, dims = [1] : (tensor<784xf32>) -> tensor<1x784xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<128x784xf32>
    %81 = stablehlo.add %80, %78 : tensor<128x784xf32>
    %82 = stablehlo.dot_general %67, %75, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x512xf32>, tensor<128x512x784xf32>) -> tensor<128x1x784xf32>
    %83 = stablehlo.broadcast_in_dim %81, dims = [0, 2] : (tensor<128x784xf32>) -> tensor<128x1x784xf32>
    %84 = stablehlo.add %82, %83 : tensor<128x1x784xf32>
    %85 = stablehlo.negate %84 : tensor<128x1x784xf32>
    %86 = stablehlo.exponential %85 : tensor<128x1x784xf32>
    %87 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1x784xf32>
    %88 = stablehlo.add %87, %86 : tensor<128x1x784xf32>
    %89 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1x784xf32>
    %90 = stablehlo.divide %89, %88 : tensor<128x1x784xf32>
    %91 = stablehlo.multiply %88, %88 : tensor<128x1x784xf32>
    %92 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1x784xf32>
    %93 = stablehlo.divide %92, %91 : tensor<128x1x784xf32>
    %94 = stablehlo.broadcast_in_dim %arg61, dims = [0, 2] : (tensor<128x784xf32>) -> tensor<128x1x784xf32>
    %95 = stablehlo.subtract %94, %90 : tensor<128x1x784xf32>
    %96 = stablehlo.multiply %95, %95 : tensor<128x1x784xf32>
    %97 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x1x784xf32>
    %98 = stablehlo.multiply %97, %95 : tensor<128x1x784xf32>
    %99 = stablehlo.reduce(%96 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<128x1x784xf32>, tensor<f32>) -> tensor<128x1xf32>
    %100 = stablehlo.reduce(%99 init: %cst_13) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x1xf32>, tensor<f32>) -> tensor<f32>
    %101 = stablehlo.divide %100, %cst_12 : tensor<f32>
    %102 = stablehlo.exponential %48 : tensor<128x1x16xf32>
    %103 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1x16xf32>
    %104 = stablehlo.add %103, %48 : tensor<128x1x16xf32>
    %105 = stablehlo.multiply %33, %33 : tensor<128x1x16xf32>
    %106 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<128x1x16xf32>
    %107 = stablehlo.multiply %106, %33 : tensor<128x1x16xf32>
    %108 = stablehlo.subtract %104, %105 : tensor<128x1x16xf32>
    %109 = stablehlo.subtract %108, %102 : tensor<128x1x16xf32>
    %110 = stablehlo.reduce(%109 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<128x1x16xf32>, tensor<f32>) -> tensor<128x1xf32>
    %111 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %112 = stablehlo.multiply %111, %110 : tensor<128x1xf32>
    %113 = stablehlo.reduce(%112 init: %cst_13) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x1xf32>, tensor<f32>) -> tensor<f32>
    %114 = stablehlo.divide %113, %cst_12 : tensor<f32>
    %115 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %116 = stablehlo.multiply %arg1, %115 : tensor<784x512xf32>
    %117 = stablehlo.log %116 : tensor<784x512xf32>
    %118 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %119 = stablehlo.add %118, %117 : tensor<784x512xf32>
    %120 = stablehlo.multiply %arg0, %arg0 : tensor<784x512xf32>
    %121 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %122 = stablehlo.multiply %121, %arg0 : tensor<784x512xf32>
    %123 = stablehlo.subtract %119, %120 : tensor<784x512xf32>
    %124 = stablehlo.subtract %123, %116 : tensor<784x512xf32>
    %125 = stablehlo.reduce(%124 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<784x512xf32>, tensor<f32>) -> tensor<784xf32>
    %126 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %127 = stablehlo.multiply %126, %125 : tensor<784xf32>
    %128 = stablehlo.reduce(%127 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<784xf32>, tensor<f32>) -> tensor<f32>
    %129 = stablehlo.divide %128, %cst_9 : tensor<f32>
    %130 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %131 = stablehlo.multiply %arg3, %130 : tensor<512xf32>
    %132 = stablehlo.log %131 : tensor<512xf32>
    %133 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %134 = stablehlo.add %133, %132 : tensor<512xf32>
    %135 = stablehlo.multiply %arg2, %arg2 : tensor<512xf32>
    %136 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %137 = stablehlo.multiply %136, %arg2 : tensor<512xf32>
    %138 = stablehlo.subtract %134, %135 : tensor<512xf32>
    %139 = stablehlo.subtract %138, %131 : tensor<512xf32>
    %140 = stablehlo.reduce(%139 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %141 = stablehlo.multiply %cst_11, %140 : tensor<f32>
    %142 = stablehlo.reduce(%141 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %143 = stablehlo.divide %142, %cst_15 : tensor<f32>
    %144 = stablehlo.add %129, %143 : tensor<f32>
    %145 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %146 = stablehlo.multiply %arg5, %145 : tensor<512x16xf32>
    %147 = stablehlo.log %146 : tensor<512x16xf32>
    %148 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %149 = stablehlo.add %148, %147 : tensor<512x16xf32>
    %150 = stablehlo.multiply %arg4, %arg4 : tensor<512x16xf32>
    %151 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %152 = stablehlo.multiply %151, %arg4 : tensor<512x16xf32>
    %153 = stablehlo.subtract %149, %150 : tensor<512x16xf32>
    %154 = stablehlo.subtract %153, %146 : tensor<512x16xf32>
    %155 = stablehlo.reduce(%154 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %156 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %157 = stablehlo.multiply %156, %155 : tensor<512xf32>
    %158 = stablehlo.reduce(%157 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %159 = stablehlo.divide %158, %cst_8 : tensor<f32>
    %160 = stablehlo.add %144, %159 : tensor<f32>
    %161 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %162 = stablehlo.multiply %arg7, %161 : tensor<16xf32>
    %163 = stablehlo.log %162 : tensor<16xf32>
    %164 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %165 = stablehlo.add %164, %163 : tensor<16xf32>
    %166 = stablehlo.multiply %arg6, %arg6 : tensor<16xf32>
    %167 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %168 = stablehlo.multiply %167, %arg6 : tensor<16xf32>
    %169 = stablehlo.subtract %165, %166 : tensor<16xf32>
    %170 = stablehlo.subtract %169, %162 : tensor<16xf32>
    %171 = stablehlo.reduce(%170 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %172 = stablehlo.multiply %cst_11, %171 : tensor<f32>
    %173 = stablehlo.reduce(%172 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %174 = stablehlo.divide %173, %cst_15 : tensor<f32>
    %175 = stablehlo.add %160, %174 : tensor<f32>
    %176 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %177 = stablehlo.multiply %arg9, %176 : tensor<512x16xf32>
    %178 = stablehlo.log %177 : tensor<512x16xf32>
    %179 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %180 = stablehlo.add %179, %178 : tensor<512x16xf32>
    %181 = stablehlo.multiply %arg8, %arg8 : tensor<512x16xf32>
    %182 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %183 = stablehlo.multiply %182, %arg8 : tensor<512x16xf32>
    %184 = stablehlo.subtract %180, %181 : tensor<512x16xf32>
    %185 = stablehlo.subtract %184, %177 : tensor<512x16xf32>
    %186 = stablehlo.reduce(%185 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %187 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %188 = stablehlo.multiply %187, %186 : tensor<512xf32>
    %189 = stablehlo.reduce(%188 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %190 = stablehlo.divide %189, %cst_8 : tensor<f32>
    %191 = stablehlo.add %175, %190 : tensor<f32>
    %192 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %193 = stablehlo.multiply %arg11, %192 : tensor<16xf32>
    %194 = stablehlo.log %193 : tensor<16xf32>
    %195 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %196 = stablehlo.add %195, %194 : tensor<16xf32>
    %197 = stablehlo.multiply %arg10, %arg10 : tensor<16xf32>
    %198 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %199 = stablehlo.multiply %198, %arg10 : tensor<16xf32>
    %200 = stablehlo.subtract %196, %197 : tensor<16xf32>
    %201 = stablehlo.subtract %200, %193 : tensor<16xf32>
    %202 = stablehlo.reduce(%201 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %203 = stablehlo.multiply %cst_11, %202 : tensor<f32>
    %204 = stablehlo.reduce(%203 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %205 = stablehlo.divide %204, %cst_15 : tensor<f32>
    %206 = stablehlo.add %191, %205 : tensor<f32>
    %207 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %208 = stablehlo.multiply %arg13, %207 : tensor<16x512xf32>
    %209 = stablehlo.log %208 : tensor<16x512xf32>
    %210 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %211 = stablehlo.add %210, %209 : tensor<16x512xf32>
    %212 = stablehlo.multiply %arg12, %arg12 : tensor<16x512xf32>
    %213 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %214 = stablehlo.multiply %213, %arg12 : tensor<16x512xf32>
    %215 = stablehlo.subtract %211, %212 : tensor<16x512xf32>
    %216 = stablehlo.subtract %215, %208 : tensor<16x512xf32>
    %217 = stablehlo.reduce(%216 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<16x512xf32>, tensor<f32>) -> tensor<16xf32>
    %218 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %219 = stablehlo.multiply %218, %217 : tensor<16xf32>
    %220 = stablehlo.reduce(%219 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %221 = stablehlo.divide %220, %cst_7 : tensor<f32>
    %222 = stablehlo.add %206, %221 : tensor<f32>
    %223 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %224 = stablehlo.multiply %arg15, %223 : tensor<512xf32>
    %225 = stablehlo.log %224 : tensor<512xf32>
    %226 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %227 = stablehlo.add %226, %225 : tensor<512xf32>
    %228 = stablehlo.multiply %arg14, %arg14 : tensor<512xf32>
    %229 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %230 = stablehlo.multiply %229, %arg14 : tensor<512xf32>
    %231 = stablehlo.subtract %227, %228 : tensor<512xf32>
    %232 = stablehlo.subtract %231, %224 : tensor<512xf32>
    %233 = stablehlo.reduce(%232 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %234 = stablehlo.multiply %cst_11, %233 : tensor<f32>
    %235 = stablehlo.reduce(%234 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %236 = stablehlo.divide %235, %cst_15 : tensor<f32>
    %237 = stablehlo.add %222, %236 : tensor<f32>
    %238 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %239 = stablehlo.multiply %arg17, %238 : tensor<512x784xf32>
    %240 = stablehlo.log %239 : tensor<512x784xf32>
    %241 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %242 = stablehlo.add %241, %240 : tensor<512x784xf32>
    %243 = stablehlo.multiply %arg16, %arg16 : tensor<512x784xf32>
    %244 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %245 = stablehlo.multiply %244, %arg16 : tensor<512x784xf32>
    %246 = stablehlo.subtract %242, %243 : tensor<512x784xf32>
    %247 = stablehlo.subtract %246, %239 : tensor<512x784xf32>
    %248 = stablehlo.reduce(%247 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<512x784xf32>, tensor<f32>) -> tensor<512xf32>
    %249 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %250 = stablehlo.multiply %249, %248 : tensor<512xf32>
    %251 = stablehlo.reduce(%250 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %252 = stablehlo.divide %251, %cst_8 : tensor<f32>
    %253 = stablehlo.add %237, %252 : tensor<f32>
    %254 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %255 = stablehlo.multiply %arg19, %254 : tensor<784xf32>
    %256 = stablehlo.log %255 : tensor<784xf32>
    %257 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %258 = stablehlo.add %257, %256 : tensor<784xf32>
    %259 = stablehlo.multiply %arg18, %arg18 : tensor<784xf32>
    %260 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %261 = stablehlo.multiply %260, %arg18 : tensor<784xf32>
    %262 = stablehlo.subtract %258, %259 : tensor<784xf32>
    %263 = stablehlo.subtract %262, %255 : tensor<784xf32>
    %264 = stablehlo.reduce(%263 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<784xf32>, tensor<f32>) -> tensor<f32>
    %265 = stablehlo.multiply %cst_11, %264 : tensor<f32>
    %266 = stablehlo.reduce(%265 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %267 = stablehlo.divide %266, %cst_15 : tensor<f32>
    %268 = stablehlo.add %253, %267 : tensor<f32>
    %269 = stablehlo.add %101, %114 : tensor<f32>
    %270 = stablehlo.add %269, %268 : tensor<f32>
    %271 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %272 = stablehlo.multiply %cst_11, %271 : tensor<f32>
    %273 = stablehlo.broadcast_in_dim %272, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %274 = stablehlo.negate %273 : tensor<784xf32>
    %275 = stablehlo.negate %273 : tensor<784xf32>
    %276 = stablehlo.multiply %275, %261 : tensor<784xf32>
    %277 = stablehlo.divide %273, %255 : tensor<784xf32>
    %278 = stablehlo.add %274, %277 : tensor<784xf32>
    %279 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %280 = stablehlo.multiply %278, %279 : tensor<784xf32>
    %281 = stablehlo.divide %cst_15, %cst_8 : tensor<f32>
    %282 = stablehlo.broadcast_in_dim %281, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %283 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %284 = stablehlo.multiply %283, %282 : tensor<512xf32>
    %285 = stablehlo.broadcast_in_dim %284, dims = [0] : (tensor<512xf32>) -> tensor<512x784xf32>
    %286 = stablehlo.negate %285 : tensor<512x784xf32>
    %287 = stablehlo.negate %285 : tensor<512x784xf32>
    %288 = stablehlo.multiply %287, %245 : tensor<512x784xf32>
    %289 = stablehlo.divide %285, %239 : tensor<512x784xf32>
    %290 = stablehlo.add %286, %289 : tensor<512x784xf32>
    %291 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %292 = stablehlo.multiply %290, %291 : tensor<512x784xf32>
    %293 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %294 = stablehlo.multiply %cst_11, %293 : tensor<f32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %296 = stablehlo.negate %295 : tensor<512xf32>
    %297 = stablehlo.negate %295 : tensor<512xf32>
    %298 = stablehlo.multiply %297, %230 : tensor<512xf32>
    %299 = stablehlo.divide %295, %224 : tensor<512xf32>
    %300 = stablehlo.add %296, %299 : tensor<512xf32>
    %301 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %302 = stablehlo.multiply %300, %301 : tensor<512xf32>
    %303 = stablehlo.divide %cst_15, %cst_7 : tensor<f32>
    %304 = stablehlo.broadcast_in_dim %303, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %305 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %306 = stablehlo.multiply %305, %304 : tensor<16xf32>
    %307 = stablehlo.broadcast_in_dim %306, dims = [0] : (tensor<16xf32>) -> tensor<16x512xf32>
    %308 = stablehlo.negate %307 : tensor<16x512xf32>
    %309 = stablehlo.negate %307 : tensor<16x512xf32>
    %310 = stablehlo.multiply %309, %214 : tensor<16x512xf32>
    %311 = stablehlo.divide %307, %208 : tensor<16x512xf32>
    %312 = stablehlo.add %308, %311 : tensor<16x512xf32>
    %313 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %314 = stablehlo.multiply %312, %313 : tensor<16x512xf32>
    %315 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %316 = stablehlo.multiply %cst_11, %315 : tensor<f32>
    %317 = stablehlo.broadcast_in_dim %316, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %318 = stablehlo.negate %317 : tensor<16xf32>
    %319 = stablehlo.negate %317 : tensor<16xf32>
    %320 = stablehlo.multiply %319, %199 : tensor<16xf32>
    %321 = stablehlo.divide %317, %193 : tensor<16xf32>
    %322 = stablehlo.add %318, %321 : tensor<16xf32>
    %323 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %324 = stablehlo.multiply %322, %323 : tensor<16xf32>
    %325 = stablehlo.divide %cst_15, %cst_8 : tensor<f32>
    %326 = stablehlo.broadcast_in_dim %325, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %327 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %328 = stablehlo.multiply %327, %326 : tensor<512xf32>
    %329 = stablehlo.broadcast_in_dim %328, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %330 = stablehlo.negate %329 : tensor<512x16xf32>
    %331 = stablehlo.negate %329 : tensor<512x16xf32>
    %332 = stablehlo.multiply %331, %183 : tensor<512x16xf32>
    %333 = stablehlo.divide %329, %177 : tensor<512x16xf32>
    %334 = stablehlo.add %330, %333 : tensor<512x16xf32>
    %335 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %336 = stablehlo.multiply %334, %335 : tensor<512x16xf32>
    %337 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %338 = stablehlo.multiply %cst_11, %337 : tensor<f32>
    %339 = stablehlo.broadcast_in_dim %338, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %340 = stablehlo.negate %339 : tensor<16xf32>
    %341 = stablehlo.negate %339 : tensor<16xf32>
    %342 = stablehlo.multiply %341, %168 : tensor<16xf32>
    %343 = stablehlo.divide %339, %162 : tensor<16xf32>
    %344 = stablehlo.add %340, %343 : tensor<16xf32>
    %345 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %346 = stablehlo.multiply %344, %345 : tensor<16xf32>
    %347 = stablehlo.divide %cst_15, %cst_8 : tensor<f32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %349 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %350 = stablehlo.multiply %349, %348 : tensor<512xf32>
    %351 = stablehlo.broadcast_in_dim %350, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %352 = stablehlo.negate %351 : tensor<512x16xf32>
    %353 = stablehlo.negate %351 : tensor<512x16xf32>
    %354 = stablehlo.multiply %353, %152 : tensor<512x16xf32>
    %355 = stablehlo.divide %351, %146 : tensor<512x16xf32>
    %356 = stablehlo.add %352, %355 : tensor<512x16xf32>
    %357 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %358 = stablehlo.multiply %356, %357 : tensor<512x16xf32>
    %359 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %360 = stablehlo.multiply %cst_11, %359 : tensor<f32>
    %361 = stablehlo.broadcast_in_dim %360, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %362 = stablehlo.negate %361 : tensor<512xf32>
    %363 = stablehlo.negate %361 : tensor<512xf32>
    %364 = stablehlo.multiply %363, %137 : tensor<512xf32>
    %365 = stablehlo.divide %361, %131 : tensor<512xf32>
    %366 = stablehlo.add %362, %365 : tensor<512xf32>
    %367 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %368 = stablehlo.multiply %366, %367 : tensor<512xf32>
    %369 = stablehlo.divide %cst_15, %cst_9 : tensor<f32>
    %370 = stablehlo.broadcast_in_dim %369, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %371 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %372 = stablehlo.multiply %371, %370 : tensor<784xf32>
    %373 = stablehlo.broadcast_in_dim %372, dims = [0] : (tensor<784xf32>) -> tensor<784x512xf32>
    %374 = stablehlo.negate %373 : tensor<784x512xf32>
    %375 = stablehlo.negate %373 : tensor<784x512xf32>
    %376 = stablehlo.multiply %375, %122 : tensor<784x512xf32>
    %377 = stablehlo.divide %373, %116 : tensor<784x512xf32>
    %378 = stablehlo.add %374, %377 : tensor<784x512xf32>
    %379 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %380 = stablehlo.multiply %378, %379 : tensor<784x512xf32>
    %381 = stablehlo.divide %cst_15, %cst_12 : tensor<f32>
    %382 = stablehlo.broadcast_in_dim %381, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %383 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %384 = stablehlo.multiply %383, %382 : tensor<128x1xf32>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x16xf32>
    %386 = stablehlo.negate %385 : tensor<128x1x16xf32>
    %387 = stablehlo.multiply %386, %102 : tensor<128x1x16xf32>
    %388 = stablehlo.negate %385 : tensor<128x1x16xf32>
    %389 = stablehlo.multiply %388, %107 : tensor<128x1x16xf32>
    %390 = stablehlo.add %387, %385 : tensor<128x1x16xf32>
    %391 = stablehlo.divide %cst_15, %cst_12 : tensor<f32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %393 = stablehlo.broadcast_in_dim %392, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x1x784xf32>
    %394 = stablehlo.multiply %393, %98 : tensor<128x1x784xf32>
    %395 = stablehlo.negate %394 : tensor<128x1x784xf32>
    %396 = stablehlo.multiply %395, %93 : tensor<128x1x784xf32>
    %397 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<128x1x784xf32>
    %398 = stablehlo.multiply %396, %397 : tensor<128x1x784xf32>
    %399 = stablehlo.negate %398 : tensor<128x1x784xf32>
    %400 = stablehlo.multiply %399, %86 : tensor<128x1x784xf32>
    %401 = stablehlo.negate %400 : tensor<128x1x784xf32>
    %402 = stablehlo.reduce(%401 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<128x1x784xf32>, tensor<f32>) -> tensor<128x784xf32>
    %403 = stablehlo.reduce(%402 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x784xf32>, tensor<f32>) -> tensor<784xf32>
    %404 = stablehlo.reshape %403 : (tensor<784xf32>) -> tensor<1x784xf32>
    %405 = stablehlo.reduce(%404 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784xf32>, tensor<f32>) -> tensor<784xf32>
    %406 = stablehlo.add %276, %405 : tensor<784xf32>
    %407 = stablehlo.multiply %402, %cst_23 : tensor<128x784xf32>
    %408 = stablehlo.reduce(%407 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x784xf32>, tensor<f32>) -> tensor<784xf32>
    %409 = stablehlo.reshape %408 : (tensor<784xf32>) -> tensor<1x784xf32>
    %410 = stablehlo.reduce(%409 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784xf32>, tensor<f32>) -> tensor<784xf32>
    %411 = stablehlo.add %280, %410 : tensor<784xf32>
    %412 = stablehlo.dot_general %401, %67, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<128x1x784xf32>, tensor<128x1x512xf32>) -> tensor<128x784x512xf32>
    %413 = stablehlo.transpose %412, dims = [0, 2, 1] : (tensor<128x784x512xf32>) -> tensor<128x512x784xf32>
    %414 = stablehlo.dot_general %401, %75, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<128x1x784xf32>, tensor<128x512x784xf32>) -> tensor<128x1x512xf32>
    %415 = stablehlo.reduce(%413 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %416 = stablehlo.reshape %415 : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %417 = stablehlo.reduce(%416 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %418 = stablehlo.add %288, %417 : tensor<512x784xf32>
    %419 = stablehlo.multiply %413, %cst_22 : tensor<128x512x784xf32>
    %420 = stablehlo.reduce(%419 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %421 = stablehlo.reshape %420 : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %422 = stablehlo.reduce(%421 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %423 = stablehlo.add %292, %422 : tensor<512x784xf32>
    %424 = stablehlo.multiply %414, %69 : tensor<128x1x512xf32>
    %425 = stablehlo.multiply %424, %67 : tensor<128x1x512xf32>
    %426 = stablehlo.add %424, %425 : tensor<128x1x512xf32>
    %427 = stablehlo.reduce(%426 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<128x1x512xf32>, tensor<f32>) -> tensor<128x512xf32>
    %428 = stablehlo.reduce(%427 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %429 = stablehlo.reshape %428 : (tensor<512xf32>) -> tensor<1x512xf32>
    %430 = stablehlo.reduce(%429 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %431 = stablehlo.add %298, %430 : tensor<512xf32>
    %432 = stablehlo.multiply %427, %cst_17 : tensor<128x512xf32>
    %433 = stablehlo.reduce(%432 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %434 = stablehlo.reshape %433 : (tensor<512xf32>) -> tensor<1x512xf32>
    %435 = stablehlo.reduce(%434 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %436 = stablehlo.add %302, %435 : tensor<512xf32>
    %437 = stablehlo.dot_general %426, %51, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<128x1x512xf32>, tensor<128x1x16xf32>) -> tensor<128x512x16xf32>
    %438 = stablehlo.transpose %437, dims = [0, 2, 1] : (tensor<128x512x16xf32>) -> tensor<128x16x512xf32>
    %439 = stablehlo.dot_general %426, %57, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<128x1x512xf32>, tensor<128x16x512xf32>) -> tensor<128x1x16xf32>
    %440 = stablehlo.reduce(%438 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %441 = stablehlo.reshape %440 : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %442 = stablehlo.reduce(%441 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %443 = stablehlo.add %310, %442 : tensor<16x512xf32>
    %444 = stablehlo.multiply %438, %cst_21 : tensor<128x16x512xf32>
    %445 = stablehlo.reduce(%444 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %446 = stablehlo.reshape %445 : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %447 = stablehlo.reduce(%446 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %448 = stablehlo.add %314, %447 : tensor<16x512xf32>
    %449 = stablehlo.add %389, %439 : tensor<128x1x16xf32>
    %450 = stablehlo.multiply %439, %cst_20 : tensor<128x1x16xf32>
    %451 = stablehlo.multiply %450, %49 : tensor<128x1x16xf32>
    %452 = stablehlo.add %390, %451 : tensor<128x1x16xf32>
    %453 = stablehlo.reduce(%452 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<128x1x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %454 = stablehlo.reduce(%453 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x16xf32>, tensor<f32>) -> tensor<16xf32>
    %455 = stablehlo.reshape %454 : (tensor<16xf32>) -> tensor<1x16xf32>
    %456 = stablehlo.reduce(%455 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %457 = stablehlo.add %320, %456 : tensor<16xf32>
    %458 = stablehlo.multiply %453, %cst_19 : tensor<128x16xf32>
    %459 = stablehlo.reduce(%458 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x16xf32>, tensor<f32>) -> tensor<16xf32>
    %460 = stablehlo.reshape %459 : (tensor<16xf32>) -> tensor<1x16xf32>
    %461 = stablehlo.reduce(%460 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %462 = stablehlo.add %324, %461 : tensor<16xf32>
    %463 = stablehlo.dot_general %452, %16, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<128x1x16xf32>, tensor<128x1x512xf32>) -> tensor<128x16x512xf32>
    %464 = stablehlo.transpose %463, dims = [0, 2, 1] : (tensor<128x16x512xf32>) -> tensor<128x512x16xf32>
    %465 = stablehlo.dot_general %452, %39, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<128x1x16xf32>, tensor<128x512x16xf32>) -> tensor<128x1x512xf32>
    %466 = stablehlo.reduce(%464 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %467 = stablehlo.reshape %466 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %468 = stablehlo.reduce(%467 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %469 = stablehlo.add %332, %468 : tensor<512x16xf32>
    %470 = stablehlo.multiply %464, %cst_18 : tensor<128x512x16xf32>
    %471 = stablehlo.reduce(%470 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %472 = stablehlo.reshape %471 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %473 = stablehlo.reduce(%472 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %474 = stablehlo.add %336, %473 : tensor<512x16xf32>
    %475 = stablehlo.reduce(%449 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<128x1x16xf32>, tensor<f32>) -> tensor<128x16xf32>
    %476 = stablehlo.reduce(%475 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x16xf32>, tensor<f32>) -> tensor<16xf32>
    %477 = stablehlo.reshape %476 : (tensor<16xf32>) -> tensor<1x16xf32>
    %478 = stablehlo.reduce(%477 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %479 = stablehlo.add %342, %478 : tensor<16xf32>
    %480 = stablehlo.multiply %475, %cst_19 : tensor<128x16xf32>
    %481 = stablehlo.reduce(%480 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x16xf32>, tensor<f32>) -> tensor<16xf32>
    %482 = stablehlo.reshape %481 : (tensor<16xf32>) -> tensor<1x16xf32>
    %483 = stablehlo.reduce(%482 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %484 = stablehlo.add %346, %483 : tensor<16xf32>
    %485 = stablehlo.dot_general %449, %16, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<128x1x16xf32>, tensor<128x1x512xf32>) -> tensor<128x16x512xf32>
    %486 = stablehlo.transpose %485, dims = [0, 2, 1] : (tensor<128x16x512xf32>) -> tensor<128x512x16xf32>
    %487 = stablehlo.dot_general %449, %24, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<128x1x16xf32>, tensor<128x512x16xf32>) -> tensor<128x1x512xf32>
    %488 = stablehlo.add %465, %487 : tensor<128x1x512xf32>
    %489 = stablehlo.reduce(%486 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %490 = stablehlo.reshape %489 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %491 = stablehlo.reduce(%490 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %492 = stablehlo.add %354, %491 : tensor<512x16xf32>
    %493 = stablehlo.multiply %486, %cst_18 : tensor<128x512x16xf32>
    %494 = stablehlo.reduce(%493 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %495 = stablehlo.reshape %494 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %496 = stablehlo.reduce(%495 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %497 = stablehlo.add %358, %496 : tensor<512x16xf32>
    %498 = stablehlo.multiply %488, %18 : tensor<128x1x512xf32>
    %499 = stablehlo.multiply %498, %16 : tensor<128x1x512xf32>
    %500 = stablehlo.add %498, %499 : tensor<128x1x512xf32>
    %501 = stablehlo.reduce(%500 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<128x1x512xf32>, tensor<f32>) -> tensor<128x512xf32>
    %502 = stablehlo.reduce(%501 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %503 = stablehlo.reshape %502 : (tensor<512xf32>) -> tensor<1x512xf32>
    %504 = stablehlo.reduce(%503 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %505 = stablehlo.add %364, %504 : tensor<512xf32>
    %506 = stablehlo.multiply %501, %cst_17 : tensor<128x512xf32>
    %507 = stablehlo.reduce(%506 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x512xf32>, tensor<f32>) -> tensor<512xf32>
    %508 = stablehlo.reshape %507 : (tensor<512xf32>) -> tensor<1x512xf32>
    %509 = stablehlo.reduce(%508 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %510 = stablehlo.add %368, %509 : tensor<512xf32>
    %511 = stablehlo.dot_general %500, %12, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<128x1x512xf32>, tensor<128x1x784xf32>) -> tensor<128x512x784xf32>
    %512 = stablehlo.transpose %511, dims = [0, 2, 1] : (tensor<128x512x784xf32>) -> tensor<128x784x512xf32>
    %513 = stablehlo.reduce(%512 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %514 = stablehlo.reshape %513 : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %515 = stablehlo.reduce(%514 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %516 = stablehlo.add %376, %515 : tensor<784x512xf32>
    %517 = stablehlo.multiply %512, %cst_16 : tensor<128x784x512xf32>
    %518 = stablehlo.reduce(%517 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<128x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %519 = stablehlo.reshape %518 : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %520 = stablehlo.reduce(%519 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %521 = stablehlo.add %380, %520 : tensor<784x512xf32>
    %522 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %523 = stablehlo.multiply %522, %516 : tensor<784x512xf32>
    %524 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %525 = stablehlo.multiply %524, %arg21 : tensor<784x512xf32>
    %526 = stablehlo.add %523, %525 : tensor<784x512xf32>
    %527 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %528 = stablehlo.multiply %527, %521 : tensor<784x512xf32>
    %529 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %530 = stablehlo.multiply %529, %arg22 : tensor<784x512xf32>
    %531 = stablehlo.add %528, %530 : tensor<784x512xf32>
    %532 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %533 = stablehlo.multiply %532, %505 : tensor<512xf32>
    %534 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %535 = stablehlo.multiply %534, %arg23 : tensor<512xf32>
    %536 = stablehlo.add %533, %535 : tensor<512xf32>
    %537 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %538 = stablehlo.multiply %537, %510 : tensor<512xf32>
    %539 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %540 = stablehlo.multiply %539, %arg24 : tensor<512xf32>
    %541 = stablehlo.add %538, %540 : tensor<512xf32>
    %542 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %543 = stablehlo.multiply %542, %492 : tensor<512x16xf32>
    %544 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %545 = stablehlo.multiply %544, %arg25 : tensor<512x16xf32>
    %546 = stablehlo.add %543, %545 : tensor<512x16xf32>
    %547 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %548 = stablehlo.multiply %547, %497 : tensor<512x16xf32>
    %549 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %550 = stablehlo.multiply %549, %arg26 : tensor<512x16xf32>
    %551 = stablehlo.add %548, %550 : tensor<512x16xf32>
    %552 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %553 = stablehlo.multiply %552, %479 : tensor<16xf32>
    %554 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %555 = stablehlo.multiply %554, %arg27 : tensor<16xf32>
    %556 = stablehlo.add %553, %555 : tensor<16xf32>
    %557 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %558 = stablehlo.multiply %557, %484 : tensor<16xf32>
    %559 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %560 = stablehlo.multiply %559, %arg28 : tensor<16xf32>
    %561 = stablehlo.add %558, %560 : tensor<16xf32>
    %562 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %563 = stablehlo.multiply %562, %469 : tensor<512x16xf32>
    %564 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %565 = stablehlo.multiply %564, %arg29 : tensor<512x16xf32>
    %566 = stablehlo.add %563, %565 : tensor<512x16xf32>
    %567 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %568 = stablehlo.multiply %567, %474 : tensor<512x16xf32>
    %569 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %570 = stablehlo.multiply %569, %arg30 : tensor<512x16xf32>
    %571 = stablehlo.add %568, %570 : tensor<512x16xf32>
    %572 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %573 = stablehlo.multiply %572, %457 : tensor<16xf32>
    %574 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %575 = stablehlo.multiply %574, %arg31 : tensor<16xf32>
    %576 = stablehlo.add %573, %575 : tensor<16xf32>
    %577 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %578 = stablehlo.multiply %577, %462 : tensor<16xf32>
    %579 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %580 = stablehlo.multiply %579, %arg32 : tensor<16xf32>
    %581 = stablehlo.add %578, %580 : tensor<16xf32>
    %582 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %583 = stablehlo.multiply %582, %443 : tensor<16x512xf32>
    %584 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %585 = stablehlo.multiply %584, %arg33 : tensor<16x512xf32>
    %586 = stablehlo.add %583, %585 : tensor<16x512xf32>
    %587 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %588 = stablehlo.multiply %587, %448 : tensor<16x512xf32>
    %589 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %590 = stablehlo.multiply %589, %arg34 : tensor<16x512xf32>
    %591 = stablehlo.add %588, %590 : tensor<16x512xf32>
    %592 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %593 = stablehlo.multiply %592, %431 : tensor<512xf32>
    %594 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %595 = stablehlo.multiply %594, %arg35 : tensor<512xf32>
    %596 = stablehlo.add %593, %595 : tensor<512xf32>
    %597 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %598 = stablehlo.multiply %597, %436 : tensor<512xf32>
    %599 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %600 = stablehlo.multiply %599, %arg36 : tensor<512xf32>
    %601 = stablehlo.add %598, %600 : tensor<512xf32>
    %602 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %603 = stablehlo.multiply %602, %418 : tensor<512x784xf32>
    %604 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %605 = stablehlo.multiply %604, %arg37 : tensor<512x784xf32>
    %606 = stablehlo.add %603, %605 : tensor<512x784xf32>
    %607 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %608 = stablehlo.multiply %607, %423 : tensor<512x784xf32>
    %609 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %610 = stablehlo.multiply %609, %arg38 : tensor<512x784xf32>
    %611 = stablehlo.add %608, %610 : tensor<512x784xf32>
    %612 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %613 = stablehlo.multiply %612, %406 : tensor<784xf32>
    %614 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %615 = stablehlo.multiply %614, %arg39 : tensor<784xf32>
    %616 = stablehlo.add %613, %615 : tensor<784xf32>
    %617 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %618 = stablehlo.multiply %617, %411 : tensor<784xf32>
    %619 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %620 = stablehlo.multiply %619, %arg40 : tensor<784xf32>
    %621 = stablehlo.add %618, %620 : tensor<784xf32>
    %622 = stablehlo.multiply %516, %516 : tensor<784x512xf32>
    %623 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %624 = stablehlo.multiply %623, %622 : tensor<784x512xf32>
    %625 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %626 = stablehlo.multiply %625, %arg41 : tensor<784x512xf32>
    %627 = stablehlo.add %624, %626 : tensor<784x512xf32>
    %628 = stablehlo.multiply %521, %521 : tensor<784x512xf32>
    %629 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %630 = stablehlo.multiply %629, %628 : tensor<784x512xf32>
    %631 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %632 = stablehlo.multiply %631, %arg42 : tensor<784x512xf32>
    %633 = stablehlo.add %630, %632 : tensor<784x512xf32>
    %634 = stablehlo.multiply %505, %505 : tensor<512xf32>
    %635 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %636 = stablehlo.multiply %635, %634 : tensor<512xf32>
    %637 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %638 = stablehlo.multiply %637, %arg43 : tensor<512xf32>
    %639 = stablehlo.add %636, %638 : tensor<512xf32>
    %640 = stablehlo.multiply %510, %510 : tensor<512xf32>
    %641 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %642 = stablehlo.multiply %641, %640 : tensor<512xf32>
    %643 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %644 = stablehlo.multiply %643, %arg44 : tensor<512xf32>
    %645 = stablehlo.add %642, %644 : tensor<512xf32>
    %646 = stablehlo.multiply %492, %492 : tensor<512x16xf32>
    %647 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %648 = stablehlo.multiply %647, %646 : tensor<512x16xf32>
    %649 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %650 = stablehlo.multiply %649, %arg45 : tensor<512x16xf32>
    %651 = stablehlo.add %648, %650 : tensor<512x16xf32>
    %652 = stablehlo.multiply %497, %497 : tensor<512x16xf32>
    %653 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %654 = stablehlo.multiply %653, %652 : tensor<512x16xf32>
    %655 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %656 = stablehlo.multiply %655, %arg46 : tensor<512x16xf32>
    %657 = stablehlo.add %654, %656 : tensor<512x16xf32>
    %658 = stablehlo.multiply %479, %479 : tensor<16xf32>
    %659 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %660 = stablehlo.multiply %659, %658 : tensor<16xf32>
    %661 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %662 = stablehlo.multiply %661, %arg47 : tensor<16xf32>
    %663 = stablehlo.add %660, %662 : tensor<16xf32>
    %664 = stablehlo.multiply %484, %484 : tensor<16xf32>
    %665 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %666 = stablehlo.multiply %665, %664 : tensor<16xf32>
    %667 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %668 = stablehlo.multiply %667, %arg48 : tensor<16xf32>
    %669 = stablehlo.add %666, %668 : tensor<16xf32>
    %670 = stablehlo.multiply %469, %469 : tensor<512x16xf32>
    %671 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %672 = stablehlo.multiply %671, %670 : tensor<512x16xf32>
    %673 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %674 = stablehlo.multiply %673, %arg49 : tensor<512x16xf32>
    %675 = stablehlo.add %672, %674 : tensor<512x16xf32>
    %676 = stablehlo.multiply %474, %474 : tensor<512x16xf32>
    %677 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %678 = stablehlo.multiply %677, %676 : tensor<512x16xf32>
    %679 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %680 = stablehlo.multiply %679, %arg50 : tensor<512x16xf32>
    %681 = stablehlo.add %678, %680 : tensor<512x16xf32>
    %682 = stablehlo.multiply %457, %457 : tensor<16xf32>
    %683 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %684 = stablehlo.multiply %683, %682 : tensor<16xf32>
    %685 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %686 = stablehlo.multiply %685, %arg51 : tensor<16xf32>
    %687 = stablehlo.add %684, %686 : tensor<16xf32>
    %688 = stablehlo.multiply %462, %462 : tensor<16xf32>
    %689 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %690 = stablehlo.multiply %689, %688 : tensor<16xf32>
    %691 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %692 = stablehlo.multiply %691, %arg52 : tensor<16xf32>
    %693 = stablehlo.add %690, %692 : tensor<16xf32>
    %694 = stablehlo.multiply %443, %443 : tensor<16x512xf32>
    %695 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %696 = stablehlo.multiply %695, %694 : tensor<16x512xf32>
    %697 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %698 = stablehlo.multiply %697, %arg53 : tensor<16x512xf32>
    %699 = stablehlo.add %696, %698 : tensor<16x512xf32>
    %700 = stablehlo.multiply %448, %448 : tensor<16x512xf32>
    %701 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %702 = stablehlo.multiply %701, %700 : tensor<16x512xf32>
    %703 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %704 = stablehlo.multiply %703, %arg54 : tensor<16x512xf32>
    %705 = stablehlo.add %702, %704 : tensor<16x512xf32>
    %706 = stablehlo.multiply %431, %431 : tensor<512xf32>
    %707 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %708 = stablehlo.multiply %707, %706 : tensor<512xf32>
    %709 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %710 = stablehlo.multiply %709, %arg55 : tensor<512xf32>
    %711 = stablehlo.add %708, %710 : tensor<512xf32>
    %712 = stablehlo.multiply %436, %436 : tensor<512xf32>
    %713 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %714 = stablehlo.multiply %713, %712 : tensor<512xf32>
    %715 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %716 = stablehlo.multiply %715, %arg56 : tensor<512xf32>
    %717 = stablehlo.add %714, %716 : tensor<512xf32>
    %718 = stablehlo.multiply %418, %418 : tensor<512x784xf32>
    %719 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %720 = stablehlo.multiply %719, %718 : tensor<512x784xf32>
    %721 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %722 = stablehlo.multiply %721, %arg57 : tensor<512x784xf32>
    %723 = stablehlo.add %720, %722 : tensor<512x784xf32>
    %724 = stablehlo.multiply %423, %423 : tensor<512x784xf32>
    %725 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %726 = stablehlo.multiply %725, %724 : tensor<512x784xf32>
    %727 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %728 = stablehlo.multiply %727, %arg58 : tensor<512x784xf32>
    %729 = stablehlo.add %726, %728 : tensor<512x784xf32>
    %730 = stablehlo.multiply %406, %406 : tensor<784xf32>
    %731 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %732 = stablehlo.multiply %731, %730 : tensor<784xf32>
    %733 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %734 = stablehlo.multiply %733, %arg59 : tensor<784xf32>
    %735 = stablehlo.add %732, %734 : tensor<784xf32>
    %736 = stablehlo.multiply %411, %411 : tensor<784xf32>
    %737 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %738 = stablehlo.multiply %737, %736 : tensor<784xf32>
    %739 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %740 = stablehlo.multiply %739, %arg60 : tensor<784xf32>
    %741 = stablehlo.add %738, %740 : tensor<784xf32>
    %742 = stablehlo.compare  LT, %arg20, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %743 = stablehlo.add %arg20, %c : tensor<i32>
    %744 = call @_where(%742, %743, %c_2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %745 = stablehlo.convert %744 : (tensor<i32>) -> tensor<f32>
    %746 = stablehlo.power %cst_5, %745 : tensor<f32>
    %747 = stablehlo.subtract %cst_15, %746 : tensor<f32>
    %748 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %749 = stablehlo.divide %526, %748 : tensor<784x512xf32>
    %750 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %751 = stablehlo.divide %531, %750 : tensor<784x512xf32>
    %752 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %753 = stablehlo.divide %536, %752 : tensor<512xf32>
    %754 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %755 = stablehlo.divide %541, %754 : tensor<512xf32>
    %756 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %757 = stablehlo.divide %546, %756 : tensor<512x16xf32>
    %758 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %759 = stablehlo.divide %551, %758 : tensor<512x16xf32>
    %760 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %761 = stablehlo.divide %556, %760 : tensor<16xf32>
    %762 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %763 = stablehlo.divide %561, %762 : tensor<16xf32>
    %764 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %765 = stablehlo.divide %566, %764 : tensor<512x16xf32>
    %766 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %767 = stablehlo.divide %571, %766 : tensor<512x16xf32>
    %768 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %769 = stablehlo.divide %576, %768 : tensor<16xf32>
    %770 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %771 = stablehlo.divide %581, %770 : tensor<16xf32>
    %772 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %773 = stablehlo.divide %586, %772 : tensor<16x512xf32>
    %774 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %775 = stablehlo.divide %591, %774 : tensor<16x512xf32>
    %776 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %777 = stablehlo.divide %596, %776 : tensor<512xf32>
    %778 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %779 = stablehlo.divide %601, %778 : tensor<512xf32>
    %780 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %781 = stablehlo.divide %606, %780 : tensor<512x784xf32>
    %782 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %783 = stablehlo.divide %611, %782 : tensor<512x784xf32>
    %784 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %785 = stablehlo.divide %616, %784 : tensor<784xf32>
    %786 = stablehlo.broadcast_in_dim %747, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %787 = stablehlo.divide %621, %786 : tensor<784xf32>
    %788 = stablehlo.convert %744 : (tensor<i32>) -> tensor<f32>
    %789 = stablehlo.power %cst_3, %788 : tensor<f32>
    %790 = stablehlo.subtract %cst_15, %789 : tensor<f32>
    %791 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %792 = stablehlo.divide %627, %791 : tensor<784x512xf32>
    %793 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %794 = stablehlo.divide %633, %793 : tensor<784x512xf32>
    %795 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %796 = stablehlo.divide %639, %795 : tensor<512xf32>
    %797 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %798 = stablehlo.divide %645, %797 : tensor<512xf32>
    %799 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %800 = stablehlo.divide %651, %799 : tensor<512x16xf32>
    %801 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %802 = stablehlo.divide %657, %801 : tensor<512x16xf32>
    %803 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %804 = stablehlo.divide %663, %803 : tensor<16xf32>
    %805 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %806 = stablehlo.divide %669, %805 : tensor<16xf32>
    %807 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %808 = stablehlo.divide %675, %807 : tensor<512x16xf32>
    %809 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %810 = stablehlo.divide %681, %809 : tensor<512x16xf32>
    %811 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %812 = stablehlo.divide %687, %811 : tensor<16xf32>
    %813 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %814 = stablehlo.divide %693, %813 : tensor<16xf32>
    %815 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %816 = stablehlo.divide %699, %815 : tensor<16x512xf32>
    %817 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %818 = stablehlo.divide %705, %817 : tensor<16x512xf32>
    %819 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %820 = stablehlo.divide %711, %819 : tensor<512xf32>
    %821 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %822 = stablehlo.divide %717, %821 : tensor<512xf32>
    %823 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %824 = stablehlo.divide %723, %823 : tensor<512x784xf32>
    %825 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %826 = stablehlo.divide %729, %825 : tensor<512x784xf32>
    %827 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %828 = stablehlo.divide %735, %827 : tensor<784xf32>
    %829 = stablehlo.broadcast_in_dim %790, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %830 = stablehlo.divide %741, %829 : tensor<784xf32>
    %831 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %832 = stablehlo.add %792, %831 : tensor<784x512xf32>
    %833 = stablehlo.sqrt %832 : tensor<784x512xf32>
    %834 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %835 = stablehlo.add %833, %834 : tensor<784x512xf32>
    %836 = stablehlo.divide %749, %835 : tensor<784x512xf32>
    %837 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %838 = stablehlo.add %794, %837 : tensor<784x512xf32>
    %839 = stablehlo.sqrt %838 : tensor<784x512xf32>
    %840 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %841 = stablehlo.add %839, %840 : tensor<784x512xf32>
    %842 = stablehlo.divide %751, %841 : tensor<784x512xf32>
    %843 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %844 = stablehlo.add %796, %843 : tensor<512xf32>
    %845 = stablehlo.sqrt %844 : tensor<512xf32>
    %846 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %847 = stablehlo.add %845, %846 : tensor<512xf32>
    %848 = stablehlo.divide %753, %847 : tensor<512xf32>
    %849 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %850 = stablehlo.add %798, %849 : tensor<512xf32>
    %851 = stablehlo.sqrt %850 : tensor<512xf32>
    %852 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %853 = stablehlo.add %851, %852 : tensor<512xf32>
    %854 = stablehlo.divide %755, %853 : tensor<512xf32>
    %855 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %856 = stablehlo.add %800, %855 : tensor<512x16xf32>
    %857 = stablehlo.sqrt %856 : tensor<512x16xf32>
    %858 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %859 = stablehlo.add %857, %858 : tensor<512x16xf32>
    %860 = stablehlo.divide %757, %859 : tensor<512x16xf32>
    %861 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %862 = stablehlo.add %802, %861 : tensor<512x16xf32>
    %863 = stablehlo.sqrt %862 : tensor<512x16xf32>
    %864 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %865 = stablehlo.add %863, %864 : tensor<512x16xf32>
    %866 = stablehlo.divide %759, %865 : tensor<512x16xf32>
    %867 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %868 = stablehlo.add %804, %867 : tensor<16xf32>
    %869 = stablehlo.sqrt %868 : tensor<16xf32>
    %870 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %871 = stablehlo.add %869, %870 : tensor<16xf32>
    %872 = stablehlo.divide %761, %871 : tensor<16xf32>
    %873 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %874 = stablehlo.add %806, %873 : tensor<16xf32>
    %875 = stablehlo.sqrt %874 : tensor<16xf32>
    %876 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %877 = stablehlo.add %875, %876 : tensor<16xf32>
    %878 = stablehlo.divide %763, %877 : tensor<16xf32>
    %879 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %880 = stablehlo.add %808, %879 : tensor<512x16xf32>
    %881 = stablehlo.sqrt %880 : tensor<512x16xf32>
    %882 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %883 = stablehlo.add %881, %882 : tensor<512x16xf32>
    %884 = stablehlo.divide %765, %883 : tensor<512x16xf32>
    %885 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %886 = stablehlo.add %810, %885 : tensor<512x16xf32>
    %887 = stablehlo.sqrt %886 : tensor<512x16xf32>
    %888 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %889 = stablehlo.add %887, %888 : tensor<512x16xf32>
    %890 = stablehlo.divide %767, %889 : tensor<512x16xf32>
    %891 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %892 = stablehlo.add %812, %891 : tensor<16xf32>
    %893 = stablehlo.sqrt %892 : tensor<16xf32>
    %894 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %895 = stablehlo.add %893, %894 : tensor<16xf32>
    %896 = stablehlo.divide %769, %895 : tensor<16xf32>
    %897 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %898 = stablehlo.add %814, %897 : tensor<16xf32>
    %899 = stablehlo.sqrt %898 : tensor<16xf32>
    %900 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %901 = stablehlo.add %899, %900 : tensor<16xf32>
    %902 = stablehlo.divide %771, %901 : tensor<16xf32>
    %903 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %904 = stablehlo.add %816, %903 : tensor<16x512xf32>
    %905 = stablehlo.sqrt %904 : tensor<16x512xf32>
    %906 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %907 = stablehlo.add %905, %906 : tensor<16x512xf32>
    %908 = stablehlo.divide %773, %907 : tensor<16x512xf32>
    %909 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %910 = stablehlo.add %818, %909 : tensor<16x512xf32>
    %911 = stablehlo.sqrt %910 : tensor<16x512xf32>
    %912 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %913 = stablehlo.add %911, %912 : tensor<16x512xf32>
    %914 = stablehlo.divide %775, %913 : tensor<16x512xf32>
    %915 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %916 = stablehlo.add %820, %915 : tensor<512xf32>
    %917 = stablehlo.sqrt %916 : tensor<512xf32>
    %918 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %919 = stablehlo.add %917, %918 : tensor<512xf32>
    %920 = stablehlo.divide %777, %919 : tensor<512xf32>
    %921 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %922 = stablehlo.add %822, %921 : tensor<512xf32>
    %923 = stablehlo.sqrt %922 : tensor<512xf32>
    %924 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %925 = stablehlo.add %923, %924 : tensor<512xf32>
    %926 = stablehlo.divide %779, %925 : tensor<512xf32>
    %927 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %928 = stablehlo.add %824, %927 : tensor<512x784xf32>
    %929 = stablehlo.sqrt %928 : tensor<512x784xf32>
    %930 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %931 = stablehlo.add %929, %930 : tensor<512x784xf32>
    %932 = stablehlo.divide %781, %931 : tensor<512x784xf32>
    %933 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %934 = stablehlo.add %826, %933 : tensor<512x784xf32>
    %935 = stablehlo.sqrt %934 : tensor<512x784xf32>
    %936 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %937 = stablehlo.add %935, %936 : tensor<512x784xf32>
    %938 = stablehlo.divide %783, %937 : tensor<512x784xf32>
    %939 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %940 = stablehlo.add %828, %939 : tensor<784xf32>
    %941 = stablehlo.sqrt %940 : tensor<784xf32>
    %942 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %943 = stablehlo.add %941, %942 : tensor<784xf32>
    %944 = stablehlo.divide %785, %943 : tensor<784xf32>
    %945 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %946 = stablehlo.add %830, %945 : tensor<784xf32>
    %947 = stablehlo.sqrt %946 : tensor<784xf32>
    %948 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %949 = stablehlo.add %947, %948 : tensor<784xf32>
    %950 = stablehlo.divide %787, %949 : tensor<784xf32>
    %951 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %952 = stablehlo.multiply %951, %arg0 : tensor<784x512xf32>
    %953 = stablehlo.add %836, %952 : tensor<784x512xf32>
    %954 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %955 = stablehlo.multiply %954, %arg1 : tensor<784x512xf32>
    %956 = stablehlo.add %842, %955 : tensor<784x512xf32>
    %957 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %958 = stablehlo.multiply %957, %arg2 : tensor<512xf32>
    %959 = stablehlo.add %848, %958 : tensor<512xf32>
    %960 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %961 = stablehlo.multiply %960, %arg3 : tensor<512xf32>
    %962 = stablehlo.add %854, %961 : tensor<512xf32>
    %963 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %964 = stablehlo.multiply %963, %arg4 : tensor<512x16xf32>
    %965 = stablehlo.add %860, %964 : tensor<512x16xf32>
    %966 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %967 = stablehlo.multiply %966, %arg5 : tensor<512x16xf32>
    %968 = stablehlo.add %866, %967 : tensor<512x16xf32>
    %969 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %970 = stablehlo.multiply %969, %arg6 : tensor<16xf32>
    %971 = stablehlo.add %872, %970 : tensor<16xf32>
    %972 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %973 = stablehlo.multiply %972, %arg7 : tensor<16xf32>
    %974 = stablehlo.add %878, %973 : tensor<16xf32>
    %975 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %976 = stablehlo.multiply %975, %arg8 : tensor<512x16xf32>
    %977 = stablehlo.add %884, %976 : tensor<512x16xf32>
    %978 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %979 = stablehlo.multiply %978, %arg9 : tensor<512x16xf32>
    %980 = stablehlo.add %890, %979 : tensor<512x16xf32>
    %981 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %982 = stablehlo.multiply %981, %arg10 : tensor<16xf32>
    %983 = stablehlo.add %896, %982 : tensor<16xf32>
    %984 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %985 = stablehlo.multiply %984, %arg11 : tensor<16xf32>
    %986 = stablehlo.add %902, %985 : tensor<16xf32>
    %987 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %988 = stablehlo.multiply %987, %arg12 : tensor<16x512xf32>
    %989 = stablehlo.add %908, %988 : tensor<16x512xf32>
    %990 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %991 = stablehlo.multiply %990, %arg13 : tensor<16x512xf32>
    %992 = stablehlo.add %914, %991 : tensor<16x512xf32>
    %993 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %994 = stablehlo.multiply %993, %arg14 : tensor<512xf32>
    %995 = stablehlo.add %920, %994 : tensor<512xf32>
    %996 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %997 = stablehlo.multiply %996, %arg15 : tensor<512xf32>
    %998 = stablehlo.add %926, %997 : tensor<512xf32>
    %999 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1000 = stablehlo.multiply %999, %arg16 : tensor<512x784xf32>
    %1001 = stablehlo.add %932, %1000 : tensor<512x784xf32>
    %1002 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1003 = stablehlo.multiply %1002, %arg17 : tensor<512x784xf32>
    %1004 = stablehlo.add %938, %1003 : tensor<512x784xf32>
    %1005 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1006 = stablehlo.multiply %1005, %arg18 : tensor<784xf32>
    %1007 = stablehlo.add %944, %1006 : tensor<784xf32>
    %1008 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1009 = stablehlo.multiply %1008, %arg19 : tensor<784xf32>
    %1010 = stablehlo.add %950, %1009 : tensor<784xf32>
    %1011 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %1012 = stablehlo.multiply %1011, %953 : tensor<784x512xf32>
    %1013 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %1014 = stablehlo.multiply %1013, %956 : tensor<784x512xf32>
    %1015 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1016 = stablehlo.multiply %1015, %959 : tensor<512xf32>
    %1017 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1018 = stablehlo.multiply %1017, %962 : tensor<512xf32>
    %1019 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1020 = stablehlo.multiply %1019, %965 : tensor<512x16xf32>
    %1021 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1022 = stablehlo.multiply %1021, %968 : tensor<512x16xf32>
    %1023 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1024 = stablehlo.multiply %1023, %971 : tensor<16xf32>
    %1025 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1026 = stablehlo.multiply %1025, %974 : tensor<16xf32>
    %1027 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1028 = stablehlo.multiply %1027, %977 : tensor<512x16xf32>
    %1029 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1030 = stablehlo.multiply %1029, %980 : tensor<512x16xf32>
    %1031 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1032 = stablehlo.multiply %1031, %983 : tensor<16xf32>
    %1033 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1034 = stablehlo.multiply %1033, %986 : tensor<16xf32>
    %1035 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1036 = stablehlo.multiply %1035, %989 : tensor<16x512xf32>
    %1037 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1038 = stablehlo.multiply %1037, %992 : tensor<16x512xf32>
    %1039 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1040 = stablehlo.multiply %1039, %995 : tensor<512xf32>
    %1041 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1042 = stablehlo.multiply %1041, %998 : tensor<512xf32>
    %1043 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1044 = stablehlo.multiply %1043, %1001 : tensor<512x784xf32>
    %1045 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1046 = stablehlo.multiply %1045, %1004 : tensor<512x784xf32>
    %1047 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1048 = stablehlo.multiply %1047, %1007 : tensor<784xf32>
    %1049 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1050 = stablehlo.multiply %1049, %1010 : tensor<784xf32>
    %1051 = stablehlo.add %arg0, %1012 : tensor<784x512xf32>
    %1052 = stablehlo.add %arg1, %1014 : tensor<784x512xf32>
    %1053 = stablehlo.add %arg2, %1016 : tensor<512xf32>
    %1054 = stablehlo.add %arg3, %1018 : tensor<512xf32>
    %1055 = stablehlo.add %arg4, %1020 : tensor<512x16xf32>
    %1056 = stablehlo.add %arg5, %1022 : tensor<512x16xf32>
    %1057 = stablehlo.add %arg6, %1024 : tensor<16xf32>
    %1058 = stablehlo.add %arg7, %1026 : tensor<16xf32>
    %1059 = stablehlo.add %arg8, %1028 : tensor<512x16xf32>
    %1060 = stablehlo.add %arg9, %1030 : tensor<512x16xf32>
    %1061 = stablehlo.add %arg10, %1032 : tensor<16xf32>
    %1062 = stablehlo.add %arg11, %1034 : tensor<16xf32>
    %1063 = stablehlo.add %arg12, %1036 : tensor<16x512xf32>
    %1064 = stablehlo.add %arg13, %1038 : tensor<16x512xf32>
    %1065 = stablehlo.add %arg14, %1040 : tensor<512xf32>
    %1066 = stablehlo.add %arg15, %1042 : tensor<512xf32>
    %1067 = stablehlo.add %arg16, %1044 : tensor<512x784xf32>
    %1068 = stablehlo.add %arg17, %1046 : tensor<512x784xf32>
    %1069 = stablehlo.add %arg18, %1048 : tensor<784xf32>
    %1070 = stablehlo.add %arg19, %1050 : tensor<784xf32>
    return %1051, %1052, %1053, %1054, %1055, %1056, %1057, %1058, %1059, %1060, %1061, %1062, %1063, %1064, %1065, %1066, %1067, %1068, %1069, %1070, %744, %526, %531, %536, %541, %546, %551, %556, %561, %566, %571, %576, %581, %586, %591, %596, %601, %606, %611, %616, %621, %627, %633, %639, %645, %651, %657, %663, %669, %675, %681, %687, %693, %699, %705, %711, %717, %723, %729, %735, %741, %270 : tensor<784x512xf32>, tensor<784x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x512xf32>, tensor<16x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x784xf32>, tensor<512x784xf32>, tensor<784xf32>, tensor<784xf32>, tensor<i32>, tensor<784x512xf32>, tensor<784x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x512xf32>, tensor<16x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x784xf32>, tensor<512x784xf32>, tensor<784xf32>, tensor<784xf32>, tensor<784x512xf32>, tensor<784x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x512xf32>, tensor<16x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x784xf32>, tensor<512x784xf32>, tensor<784xf32>, tensor<784xf32>, tensor<f32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
}
