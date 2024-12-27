module @jit_optim attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xui32>, %arg1: tensor<784x512xf32>, %arg2: tensor<784x512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512x16xf32>, %arg6: tensor<512x16xf32>, %arg7: tensor<16xf32>, %arg8: tensor<16xf32>, %arg9: tensor<512x16xf32>, %arg10: tensor<512x16xf32>, %arg11: tensor<16xf32>, %arg12: tensor<16xf32>, %arg13: tensor<16x512xf32>, %arg14: tensor<16x512xf32>, %arg15: tensor<512xf32>, %arg16: tensor<512xf32>, %arg17: tensor<512x784xf32>, %arg18: tensor<512x784xf32>, %arg19: tensor<784xf32>, %arg20: tensor<784xf32>, %arg21: tensor<i32>, %arg22: tensor<784x512xf32>, %arg23: tensor<784x512xf32>, %arg24: tensor<512xf32>, %arg25: tensor<512xf32>, %arg26: tensor<512x16xf32>, %arg27: tensor<512x16xf32>, %arg28: tensor<16xf32>, %arg29: tensor<16xf32>, %arg30: tensor<512x16xf32>, %arg31: tensor<512x16xf32>, %arg32: tensor<16xf32>, %arg33: tensor<16xf32>, %arg34: tensor<16x512xf32>, %arg35: tensor<16x512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512xf32>, %arg38: tensor<512x784xf32>, %arg39: tensor<512x784xf32>, %arg40: tensor<784xf32>, %arg41: tensor<784xf32>, %arg42: tensor<784x512xf32>, %arg43: tensor<784x512xf32>, %arg44: tensor<512xf32>, %arg45: tensor<512xf32>, %arg46: tensor<512x16xf32>, %arg47: tensor<512x16xf32>, %arg48: tensor<16xf32>, %arg49: tensor<16xf32>, %arg50: tensor<512x16xf32>, %arg51: tensor<512x16xf32>, %arg52: tensor<16xf32>, %arg53: tensor<16xf32>, %arg54: tensor<16x512xf32>, %arg55: tensor<16x512xf32>, %arg56: tensor<512xf32>, %arg57: tensor<512xf32>, %arg58: tensor<512x784xf32>, %arg59: tensor<512x784xf32>, %arg60: tensor<784xf32>, %arg61: tensor<784xf32>, %arg62: tensor<32x784xf32>) -> (tensor<784x512xf32> {jax.result_info = "[0][0][0]"}, tensor<784x512xf32> {jax.result_info = "[0][0][1]"}, tensor<512xf32> {jax.result_info = "[0][1][0]"}, tensor<512xf32> {jax.result_info = "[0][1][1]"}, tensor<512x16xf32> {jax.result_info = "[0][2][0]"}, tensor<512x16xf32> {jax.result_info = "[0][2][1]"}, tensor<16xf32> {jax.result_info = "[0][3][0]"}, tensor<16xf32> {jax.result_info = "[0][3][1]"}, tensor<512x16xf32> {jax.result_info = "[0][4][0]"}, tensor<512x16xf32> {jax.result_info = "[0][4][1]"}, tensor<16xf32> {jax.result_info = "[0][5][0]"}, tensor<16xf32> {jax.result_info = "[0][5][1]"}, tensor<16x512xf32> {jax.result_info = "[0][6][0]"}, tensor<16x512xf32> {jax.result_info = "[0][6][1]"}, tensor<512xf32> {jax.result_info = "[0][7][0]"}, tensor<512xf32> {jax.result_info = "[0][7][1]"}, tensor<512x784xf32> {jax.result_info = "[0][8][0]"}, tensor<512x784xf32> {jax.result_info = "[0][8][1]"}, tensor<784xf32> {jax.result_info = "[0][9][0]"}, tensor<784xf32> {jax.result_info = "[0][9][1]"}, tensor<i32> {jax.result_info = "[0][10][0].count"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].mu[0][0]"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].mu[0][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[1][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[1][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[2][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[2][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[3][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[3][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[4][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].mu[4][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[5][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].mu[5][1]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].mu[6][0]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].mu[6][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[7][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].mu[7][1]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].mu[8][0]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].mu[8][1]"}, tensor<784xf32> {jax.result_info = "[0][10][0].mu[9][0]"}, tensor<784xf32> {jax.result_info = "[0][10][0].mu[9][1]"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].nu[0][0]"}, tensor<784x512xf32> {jax.result_info = "[0][10][0].nu[0][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[1][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[1][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[2][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[2][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[3][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[3][1]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[4][0]"}, tensor<512x16xf32> {jax.result_info = "[0][10][0].nu[4][1]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[5][0]"}, tensor<16xf32> {jax.result_info = "[0][10][0].nu[5][1]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].nu[6][0]"}, tensor<16x512xf32> {jax.result_info = "[0][10][0].nu[6][1]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[7][0]"}, tensor<512xf32> {jax.result_info = "[0][10][0].nu[7][1]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].nu[8][0]"}, tensor<512x784xf32> {jax.result_info = "[0][10][0].nu[8][1]"}, tensor<784xf32> {jax.result_info = "[0][10][0].nu[9][0]"}, tensor<784xf32> {jax.result_info = "[0][10][0].nu[9][1]"}, tensor<f32> {jax.result_info = "[1]"}) {
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
    %cst_12 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_14 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = call @_threefry_split(%arg0) : (tensor<2xui32>) -> tensor<2x2xui32>
    %2 = stablehlo.slice %1 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %3 = stablehlo.reshape %2 : (tensor<1x2xui32>) -> tensor<2xui32>
    %4 = stablehlo.slice %1 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %5 = stablehlo.reshape %4 : (tensor<1x2xui32>) -> tensor<2xui32>
    %6 = call @_normal(%3) : (tensor<2xui32>) -> tensor<32x784x512xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [1, 2] : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x784x512xf32>) -> tensor<32x784x512xf32>
    %9 = stablehlo.multiply %8, %6 : tensor<32x784x512xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x784x512xf32>) -> tensor<32x784x512xf32>
    %12 = stablehlo.add %11, %9 : tensor<32x784x512xf32>
    %13 = call @_threefry_split(%5) : (tensor<2xui32>) -> tensor<2x2xui32>
    %14 = stablehlo.slice %13 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %15 = stablehlo.reshape %14 : (tensor<1x2xui32>) -> tensor<2xui32>
    %16 = stablehlo.slice %13 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %17 = stablehlo.reshape %16 : (tensor<1x2xui32>) -> tensor<2xui32>
    %18 = call @_normal_1(%15) : (tensor<2xui32>) -> tensor<32x512xf32>
    %19 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<32x512xf32>
    %21 = stablehlo.multiply %20, %18 : tensor<32x512xf32>
    %22 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<32x512xf32>
    %24 = stablehlo.add %23, %21 : tensor<32x512xf32>
    %25 = stablehlo.broadcast_in_dim %arg62, dims = [0, 2] : (tensor<32x784xf32>) -> tensor<32x1x784xf32>
    %26 = stablehlo.dot_general %25, %12, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x1x784xf32>, tensor<32x784x512xf32>) -> tensor<32x1x512xf32>
    %27 = stablehlo.broadcast_in_dim %24, dims = [0, 2] : (tensor<32x512xf32>) -> tensor<32x1x512xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x1x512xf32>
    %29 = stablehlo.tanh %28 : tensor<32x1x512xf32>
    %30 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<32x1x512xf32>
    %31 = stablehlo.subtract %30, %29 : tensor<32x1x512xf32>
    %32 = call @_threefry_split(%17) : (tensor<2xui32>) -> tensor<2x2xui32>
    %33 = stablehlo.slice %32 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %34 = stablehlo.reshape %33 : (tensor<1x2xui32>) -> tensor<2xui32>
    %35 = stablehlo.slice %32 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %36 = stablehlo.reshape %35 : (tensor<1x2xui32>) -> tensor<2xui32>
    %37 = call @_normal_5(%34) : (tensor<2xui32>) -> tensor<32x512x16xf32>
    %38 = stablehlo.broadcast_in_dim %arg6, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<32x512x16xf32>
    %40 = stablehlo.multiply %39, %37 : tensor<32x512x16xf32>
    %41 = stablehlo.broadcast_in_dim %arg5, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<32x512x16xf32>
    %43 = stablehlo.add %42, %40 : tensor<32x512x16xf32>
    %44 = call @_threefry_split(%36) : (tensor<2xui32>) -> tensor<2x2xui32>
    %45 = stablehlo.slice %44 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %46 = stablehlo.reshape %45 : (tensor<1x2xui32>) -> tensor<2xui32>
    %47 = stablehlo.slice %44 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %48 = stablehlo.reshape %47 : (tensor<1x2xui32>) -> tensor<2xui32>
    %49 = call @_normal_9(%46) : (tensor<2xui32>) -> tensor<32x16xf32>
    %50 = stablehlo.broadcast_in_dim %arg8, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %52 = stablehlo.multiply %51, %49 : tensor<32x16xf32>
    %53 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %55 = stablehlo.add %54, %52 : tensor<32x16xf32>
    %56 = stablehlo.dot_general %29, %43, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x1x512xf32>, tensor<32x512x16xf32>) -> tensor<32x1x16xf32>
    %57 = stablehlo.broadcast_in_dim %55, dims = [0, 2] : (tensor<32x16xf32>) -> tensor<32x1x16xf32>
    %58 = stablehlo.add %56, %57 : tensor<32x1x16xf32>
    %59 = call @_threefry_split(%48) : (tensor<2xui32>) -> tensor<2x2xui32>
    %60 = stablehlo.slice %59 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %61 = stablehlo.reshape %60 : (tensor<1x2xui32>) -> tensor<2xui32>
    %62 = stablehlo.slice %59 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %63 = stablehlo.reshape %62 : (tensor<1x2xui32>) -> tensor<2xui32>
    %64 = call @_normal_5(%61) : (tensor<2xui32>) -> tensor<32x512x16xf32>
    %65 = stablehlo.broadcast_in_dim %arg10, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<32x512x16xf32>
    %67 = stablehlo.multiply %66, %64 : tensor<32x512x16xf32>
    %68 = stablehlo.broadcast_in_dim %arg9, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<32x512x16xf32>
    %70 = stablehlo.add %69, %67 : tensor<32x512x16xf32>
    %71 = call @_threefry_split(%63) : (tensor<2xui32>) -> tensor<2x2xui32>
    %72 = stablehlo.slice %71 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %73 = stablehlo.reshape %72 : (tensor<1x2xui32>) -> tensor<2xui32>
    %74 = stablehlo.slice %71 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %75 = stablehlo.reshape %74 : (tensor<1x2xui32>) -> tensor<2xui32>
    %76 = call @_normal_9(%73) : (tensor<2xui32>) -> tensor<32x16xf32>
    %77 = stablehlo.broadcast_in_dim %arg12, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %79 = stablehlo.multiply %78, %76 : tensor<32x16xf32>
    %80 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %82 = stablehlo.add %81, %79 : tensor<32x16xf32>
    %83 = stablehlo.dot_general %29, %70, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x1x512xf32>, tensor<32x512x16xf32>) -> tensor<32x1x16xf32>
    %84 = stablehlo.broadcast_in_dim %82, dims = [0, 2] : (tensor<32x16xf32>) -> tensor<32x1x16xf32>
    %85 = stablehlo.add %83, %84 : tensor<32x1x16xf32>
    %86 = stablehlo.exponential %85 : tensor<32x1x16xf32>
    %87 = call @_threefry_split(%75) : (tensor<2xui32>) -> tensor<2x2xui32>
    %88 = stablehlo.slice %87 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %89 = stablehlo.reshape %88 : (tensor<1x2xui32>) -> tensor<2xui32>
    %90 = stablehlo.slice %87 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %91 = stablehlo.reshape %90 : (tensor<1x2xui32>) -> tensor<2xui32>
    %92 = call @_normal_13(%89) : (tensor<2xui32>) -> tensor<32x1x16xf32>
    %93 = stablehlo.multiply %86, %92 : tensor<32x1x16xf32>
    %94 = stablehlo.add %58, %93 : tensor<32x1x16xf32>
    %95 = call @_threefry_split(%91) : (tensor<2xui32>) -> tensor<2x2xui32>
    %96 = stablehlo.slice %95 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %97 = stablehlo.reshape %96 : (tensor<1x2xui32>) -> tensor<2xui32>
    %98 = stablehlo.slice %95 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %99 = stablehlo.reshape %98 : (tensor<1x2xui32>) -> tensor<2xui32>
    %100 = call @_normal_16(%97) : (tensor<2xui32>) -> tensor<32x16x512xf32>
    %101 = stablehlo.broadcast_in_dim %arg14, dims = [1, 2] : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1, 2] : (tensor<1x16x512xf32>) -> tensor<32x16x512xf32>
    %103 = stablehlo.multiply %102, %100 : tensor<32x16x512xf32>
    %104 = stablehlo.broadcast_in_dim %arg13, dims = [1, 2] : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2] : (tensor<1x16x512xf32>) -> tensor<32x16x512xf32>
    %106 = stablehlo.add %105, %103 : tensor<32x16x512xf32>
    %107 = call @_threefry_split(%99) : (tensor<2xui32>) -> tensor<2x2xui32>
    %108 = stablehlo.slice %107 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %109 = stablehlo.reshape %108 : (tensor<1x2xui32>) -> tensor<2xui32>
    %110 = stablehlo.slice %107 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %111 = stablehlo.reshape %110 : (tensor<1x2xui32>) -> tensor<2xui32>
    %112 = call @_normal_1(%109) : (tensor<2xui32>) -> tensor<32x512xf32>
    %113 = stablehlo.broadcast_in_dim %arg16, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<32x512xf32>
    %115 = stablehlo.multiply %114, %112 : tensor<32x512xf32>
    %116 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<32x512xf32>
    %118 = stablehlo.add %117, %115 : tensor<32x512xf32>
    %119 = stablehlo.dot_general %94, %106, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x1x16xf32>, tensor<32x16x512xf32>) -> tensor<32x1x512xf32>
    %120 = stablehlo.broadcast_in_dim %118, dims = [0, 2] : (tensor<32x512xf32>) -> tensor<32x1x512xf32>
    %121 = stablehlo.add %119, %120 : tensor<32x1x512xf32>
    %122 = stablehlo.tanh %121 : tensor<32x1x512xf32>
    %123 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<32x1x512xf32>
    %124 = stablehlo.subtract %123, %122 : tensor<32x1x512xf32>
    %125 = call @_threefry_split(%111) : (tensor<2xui32>) -> tensor<2x2xui32>
    %126 = stablehlo.slice %125 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %127 = stablehlo.reshape %126 : (tensor<1x2xui32>) -> tensor<2xui32>
    %128 = stablehlo.slice %125 [1:2, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %129 = stablehlo.reshape %128 : (tensor<1x2xui32>) -> tensor<2xui32>
    %130 = call @_normal_19(%127) : (tensor<2xui32>) -> tensor<32x512x784xf32>
    %131 = stablehlo.broadcast_in_dim %arg18, dims = [1, 2] : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2] : (tensor<1x512x784xf32>) -> tensor<32x512x784xf32>
    %133 = stablehlo.multiply %132, %130 : tensor<32x512x784xf32>
    %134 = stablehlo.broadcast_in_dim %arg17, dims = [1, 2] : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %135 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2] : (tensor<1x512x784xf32>) -> tensor<32x512x784xf32>
    %136 = stablehlo.add %135, %133 : tensor<32x512x784xf32>
    %137 = call @_threefry_split(%129) : (tensor<2xui32>) -> tensor<2x2xui32>
    %138 = stablehlo.slice %137 [0:1, 0:2] : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %139 = stablehlo.reshape %138 : (tensor<1x2xui32>) -> tensor<2xui32>
    %140 = call @_normal_22(%139) : (tensor<2xui32>) -> tensor<32x784xf32>
    %141 = stablehlo.broadcast_in_dim %arg20, dims = [1] : (tensor<784xf32>) -> tensor<1x784xf32>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<32x784xf32>
    %143 = stablehlo.multiply %142, %140 : tensor<32x784xf32>
    %144 = stablehlo.broadcast_in_dim %arg19, dims = [1] : (tensor<784xf32>) -> tensor<1x784xf32>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<32x784xf32>
    %146 = stablehlo.add %145, %143 : tensor<32x784xf32>
    %147 = stablehlo.dot_general %122, %136, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x1x512xf32>, tensor<32x512x784xf32>) -> tensor<32x1x784xf32>
    %148 = stablehlo.broadcast_in_dim %146, dims = [0, 2] : (tensor<32x784xf32>) -> tensor<32x1x784xf32>
    %149 = stablehlo.add %147, %148 : tensor<32x1x784xf32>
    %150 = stablehlo.negate %149 : tensor<32x1x784xf32>
    %151 = stablehlo.exponential %150 : tensor<32x1x784xf32>
    %152 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<32x1x784xf32>
    %153 = stablehlo.add %152, %151 : tensor<32x1x784xf32>
    %154 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<32x1x784xf32>
    %155 = stablehlo.divide %154, %153 : tensor<32x1x784xf32>
    %156 = stablehlo.multiply %153, %153 : tensor<32x1x784xf32>
    %157 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<32x1x784xf32>
    %158 = stablehlo.divide %157, %156 : tensor<32x1x784xf32>
    %159 = stablehlo.broadcast_in_dim %arg62, dims = [0, 2] : (tensor<32x784xf32>) -> tensor<32x1x784xf32>
    %160 = stablehlo.subtract %159, %155 : tensor<32x1x784xf32>
    %161 = stablehlo.multiply %160, %160 : tensor<32x1x784xf32>
    %162 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<32x1x784xf32>
    %163 = stablehlo.multiply %162, %160 : tensor<32x1x784xf32>
    %164 = stablehlo.reduce(%161 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<32x1x784xf32>, tensor<f32>) -> tensor<32x1xf32>
    %165 = stablehlo.reduce(%164 init: %cst_13) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x1xf32>, tensor<f32>) -> tensor<f32>
    %166 = stablehlo.divide %165, %cst_12 : tensor<f32>
    %167 = stablehlo.exponential %85 : tensor<32x1x16xf32>
    %168 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<32x1x16xf32>
    %169 = stablehlo.add %168, %85 : tensor<32x1x16xf32>
    %170 = stablehlo.multiply %58, %58 : tensor<32x1x16xf32>
    %171 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<32x1x16xf32>
    %172 = stablehlo.multiply %171, %58 : tensor<32x1x16xf32>
    %173 = stablehlo.subtract %169, %170 : tensor<32x1x16xf32>
    %174 = stablehlo.subtract %173, %167 : tensor<32x1x16xf32>
    %175 = stablehlo.reduce(%174 init: %cst_13) applies stablehlo.add across dimensions = [2] : (tensor<32x1x16xf32>, tensor<f32>) -> tensor<32x1xf32>
    %176 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %177 = stablehlo.multiply %176, %175 : tensor<32x1xf32>
    %178 = stablehlo.reduce(%177 init: %cst_13) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x1xf32>, tensor<f32>) -> tensor<f32>
    %179 = stablehlo.divide %178, %cst_12 : tensor<f32>
    %180 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %181 = stablehlo.multiply %arg2, %180 : tensor<784x512xf32>
    %182 = stablehlo.log %181 : tensor<784x512xf32>
    %183 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %184 = stablehlo.add %183, %182 : tensor<784x512xf32>
    %185 = stablehlo.multiply %arg1, %arg1 : tensor<784x512xf32>
    %186 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %187 = stablehlo.multiply %186, %arg1 : tensor<784x512xf32>
    %188 = stablehlo.subtract %184, %185 : tensor<784x512xf32>
    %189 = stablehlo.subtract %188, %181 : tensor<784x512xf32>
    %190 = stablehlo.reduce(%189 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<784x512xf32>, tensor<f32>) -> tensor<784xf32>
    %191 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %192 = stablehlo.multiply %191, %190 : tensor<784xf32>
    %193 = stablehlo.reduce(%192 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<784xf32>, tensor<f32>) -> tensor<f32>
    %194 = stablehlo.divide %193, %cst_9 : tensor<f32>
    %195 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %196 = stablehlo.multiply %arg4, %195 : tensor<512xf32>
    %197 = stablehlo.log %196 : tensor<512xf32>
    %198 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %199 = stablehlo.add %198, %197 : tensor<512xf32>
    %200 = stablehlo.multiply %arg3, %arg3 : tensor<512xf32>
    %201 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %202 = stablehlo.multiply %201, %arg3 : tensor<512xf32>
    %203 = stablehlo.subtract %199, %200 : tensor<512xf32>
    %204 = stablehlo.subtract %203, %196 : tensor<512xf32>
    %205 = stablehlo.reduce(%204 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %206 = stablehlo.multiply %cst_11, %205 : tensor<f32>
    %207 = stablehlo.reduce(%206 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %208 = stablehlo.divide %207, %cst_15 : tensor<f32>
    %209 = stablehlo.add %194, %208 : tensor<f32>
    %210 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %211 = stablehlo.multiply %arg6, %210 : tensor<512x16xf32>
    %212 = stablehlo.log %211 : tensor<512x16xf32>
    %213 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %214 = stablehlo.add %213, %212 : tensor<512x16xf32>
    %215 = stablehlo.multiply %arg5, %arg5 : tensor<512x16xf32>
    %216 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %217 = stablehlo.multiply %216, %arg5 : tensor<512x16xf32>
    %218 = stablehlo.subtract %214, %215 : tensor<512x16xf32>
    %219 = stablehlo.subtract %218, %211 : tensor<512x16xf32>
    %220 = stablehlo.reduce(%219 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %221 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %222 = stablehlo.multiply %221, %220 : tensor<512xf32>
    %223 = stablehlo.reduce(%222 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %224 = stablehlo.divide %223, %cst_8 : tensor<f32>
    %225 = stablehlo.add %209, %224 : tensor<f32>
    %226 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %227 = stablehlo.multiply %arg8, %226 : tensor<16xf32>
    %228 = stablehlo.log %227 : tensor<16xf32>
    %229 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %230 = stablehlo.add %229, %228 : tensor<16xf32>
    %231 = stablehlo.multiply %arg7, %arg7 : tensor<16xf32>
    %232 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %233 = stablehlo.multiply %232, %arg7 : tensor<16xf32>
    %234 = stablehlo.subtract %230, %231 : tensor<16xf32>
    %235 = stablehlo.subtract %234, %227 : tensor<16xf32>
    %236 = stablehlo.reduce(%235 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %237 = stablehlo.multiply %cst_11, %236 : tensor<f32>
    %238 = stablehlo.reduce(%237 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %239 = stablehlo.divide %238, %cst_15 : tensor<f32>
    %240 = stablehlo.add %225, %239 : tensor<f32>
    %241 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %242 = stablehlo.multiply %arg10, %241 : tensor<512x16xf32>
    %243 = stablehlo.log %242 : tensor<512x16xf32>
    %244 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %245 = stablehlo.add %244, %243 : tensor<512x16xf32>
    %246 = stablehlo.multiply %arg9, %arg9 : tensor<512x16xf32>
    %247 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %248 = stablehlo.multiply %247, %arg9 : tensor<512x16xf32>
    %249 = stablehlo.subtract %245, %246 : tensor<512x16xf32>
    %250 = stablehlo.subtract %249, %242 : tensor<512x16xf32>
    %251 = stablehlo.reduce(%250 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %252 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %253 = stablehlo.multiply %252, %251 : tensor<512xf32>
    %254 = stablehlo.reduce(%253 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %255 = stablehlo.divide %254, %cst_8 : tensor<f32>
    %256 = stablehlo.add %240, %255 : tensor<f32>
    %257 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %258 = stablehlo.multiply %arg12, %257 : tensor<16xf32>
    %259 = stablehlo.log %258 : tensor<16xf32>
    %260 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %261 = stablehlo.add %260, %259 : tensor<16xf32>
    %262 = stablehlo.multiply %arg11, %arg11 : tensor<16xf32>
    %263 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %264 = stablehlo.multiply %263, %arg11 : tensor<16xf32>
    %265 = stablehlo.subtract %261, %262 : tensor<16xf32>
    %266 = stablehlo.subtract %265, %258 : tensor<16xf32>
    %267 = stablehlo.reduce(%266 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %268 = stablehlo.multiply %cst_11, %267 : tensor<f32>
    %269 = stablehlo.reduce(%268 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %270 = stablehlo.divide %269, %cst_15 : tensor<f32>
    %271 = stablehlo.add %256, %270 : tensor<f32>
    %272 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %273 = stablehlo.multiply %arg14, %272 : tensor<16x512xf32>
    %274 = stablehlo.log %273 : tensor<16x512xf32>
    %275 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %276 = stablehlo.add %275, %274 : tensor<16x512xf32>
    %277 = stablehlo.multiply %arg13, %arg13 : tensor<16x512xf32>
    %278 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %279 = stablehlo.multiply %278, %arg13 : tensor<16x512xf32>
    %280 = stablehlo.subtract %276, %277 : tensor<16x512xf32>
    %281 = stablehlo.subtract %280, %273 : tensor<16x512xf32>
    %282 = stablehlo.reduce(%281 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<16x512xf32>, tensor<f32>) -> tensor<16xf32>
    %283 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %284 = stablehlo.multiply %283, %282 : tensor<16xf32>
    %285 = stablehlo.reduce(%284 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %286 = stablehlo.divide %285, %cst_7 : tensor<f32>
    %287 = stablehlo.add %271, %286 : tensor<f32>
    %288 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %289 = stablehlo.multiply %arg16, %288 : tensor<512xf32>
    %290 = stablehlo.log %289 : tensor<512xf32>
    %291 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %292 = stablehlo.add %291, %290 : tensor<512xf32>
    %293 = stablehlo.multiply %arg15, %arg15 : tensor<512xf32>
    %294 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %295 = stablehlo.multiply %294, %arg15 : tensor<512xf32>
    %296 = stablehlo.subtract %292, %293 : tensor<512xf32>
    %297 = stablehlo.subtract %296, %289 : tensor<512xf32>
    %298 = stablehlo.reduce(%297 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %299 = stablehlo.multiply %cst_11, %298 : tensor<f32>
    %300 = stablehlo.reduce(%299 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %301 = stablehlo.divide %300, %cst_15 : tensor<f32>
    %302 = stablehlo.add %287, %301 : tensor<f32>
    %303 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %304 = stablehlo.multiply %arg18, %303 : tensor<512x784xf32>
    %305 = stablehlo.log %304 : tensor<512x784xf32>
    %306 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %307 = stablehlo.add %306, %305 : tensor<512x784xf32>
    %308 = stablehlo.multiply %arg17, %arg17 : tensor<512x784xf32>
    %309 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %310 = stablehlo.multiply %309, %arg17 : tensor<512x784xf32>
    %311 = stablehlo.subtract %307, %308 : tensor<512x784xf32>
    %312 = stablehlo.subtract %311, %304 : tensor<512x784xf32>
    %313 = stablehlo.reduce(%312 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<512x784xf32>, tensor<f32>) -> tensor<512xf32>
    %314 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %315 = stablehlo.multiply %314, %313 : tensor<512xf32>
    %316 = stablehlo.reduce(%315 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %317 = stablehlo.divide %316, %cst_8 : tensor<f32>
    %318 = stablehlo.add %302, %317 : tensor<f32>
    %319 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %320 = stablehlo.multiply %arg20, %319 : tensor<784xf32>
    %321 = stablehlo.log %320 : tensor<784xf32>
    %322 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %323 = stablehlo.add %322, %321 : tensor<784xf32>
    %324 = stablehlo.multiply %arg19, %arg19 : tensor<784xf32>
    %325 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %326 = stablehlo.multiply %325, %arg19 : tensor<784xf32>
    %327 = stablehlo.subtract %323, %324 : tensor<784xf32>
    %328 = stablehlo.subtract %327, %320 : tensor<784xf32>
    %329 = stablehlo.reduce(%328 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<784xf32>, tensor<f32>) -> tensor<f32>
    %330 = stablehlo.multiply %cst_11, %329 : tensor<f32>
    %331 = stablehlo.reduce(%330 init: %cst_13) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %332 = stablehlo.divide %331, %cst_15 : tensor<f32>
    %333 = stablehlo.add %318, %332 : tensor<f32>
    %334 = stablehlo.add %166, %179 : tensor<f32>
    %335 = stablehlo.add %334, %333 : tensor<f32>
    %336 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %337 = stablehlo.multiply %cst_11, %336 : tensor<f32>
    %338 = stablehlo.broadcast_in_dim %337, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %339 = stablehlo.negate %338 : tensor<784xf32>
    %340 = stablehlo.negate %338 : tensor<784xf32>
    %341 = stablehlo.multiply %340, %326 : tensor<784xf32>
    %342 = stablehlo.divide %338, %320 : tensor<784xf32>
    %343 = stablehlo.add %339, %342 : tensor<784xf32>
    %344 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %345 = stablehlo.multiply %343, %344 : tensor<784xf32>
    %346 = stablehlo.divide %cst_15, %cst_8 : tensor<f32>
    %347 = stablehlo.broadcast_in_dim %346, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %348 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %349 = stablehlo.multiply %348, %347 : tensor<512xf32>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0] : (tensor<512xf32>) -> tensor<512x784xf32>
    %351 = stablehlo.negate %350 : tensor<512x784xf32>
    %352 = stablehlo.negate %350 : tensor<512x784xf32>
    %353 = stablehlo.multiply %352, %310 : tensor<512x784xf32>
    %354 = stablehlo.divide %350, %304 : tensor<512x784xf32>
    %355 = stablehlo.add %351, %354 : tensor<512x784xf32>
    %356 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %357 = stablehlo.multiply %355, %356 : tensor<512x784xf32>
    %358 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %359 = stablehlo.multiply %cst_11, %358 : tensor<f32>
    %360 = stablehlo.broadcast_in_dim %359, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %361 = stablehlo.negate %360 : tensor<512xf32>
    %362 = stablehlo.negate %360 : tensor<512xf32>
    %363 = stablehlo.multiply %362, %295 : tensor<512xf32>
    %364 = stablehlo.divide %360, %289 : tensor<512xf32>
    %365 = stablehlo.add %361, %364 : tensor<512xf32>
    %366 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %367 = stablehlo.multiply %365, %366 : tensor<512xf32>
    %368 = stablehlo.divide %cst_15, %cst_7 : tensor<f32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %370 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %371 = stablehlo.multiply %370, %369 : tensor<16xf32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0] : (tensor<16xf32>) -> tensor<16x512xf32>
    %373 = stablehlo.negate %372 : tensor<16x512xf32>
    %374 = stablehlo.negate %372 : tensor<16x512xf32>
    %375 = stablehlo.multiply %374, %279 : tensor<16x512xf32>
    %376 = stablehlo.divide %372, %273 : tensor<16x512xf32>
    %377 = stablehlo.add %373, %376 : tensor<16x512xf32>
    %378 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %379 = stablehlo.multiply %377, %378 : tensor<16x512xf32>
    %380 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %381 = stablehlo.multiply %cst_11, %380 : tensor<f32>
    %382 = stablehlo.broadcast_in_dim %381, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %383 = stablehlo.negate %382 : tensor<16xf32>
    %384 = stablehlo.negate %382 : tensor<16xf32>
    %385 = stablehlo.multiply %384, %264 : tensor<16xf32>
    %386 = stablehlo.divide %382, %258 : tensor<16xf32>
    %387 = stablehlo.add %383, %386 : tensor<16xf32>
    %388 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %389 = stablehlo.multiply %387, %388 : tensor<16xf32>
    %390 = stablehlo.divide %cst_15, %cst_8 : tensor<f32>
    %391 = stablehlo.broadcast_in_dim %390, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %392 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %393 = stablehlo.multiply %392, %391 : tensor<512xf32>
    %394 = stablehlo.broadcast_in_dim %393, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %395 = stablehlo.negate %394 : tensor<512x16xf32>
    %396 = stablehlo.negate %394 : tensor<512x16xf32>
    %397 = stablehlo.multiply %396, %248 : tensor<512x16xf32>
    %398 = stablehlo.divide %394, %242 : tensor<512x16xf32>
    %399 = stablehlo.add %395, %398 : tensor<512x16xf32>
    %400 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %401 = stablehlo.multiply %399, %400 : tensor<512x16xf32>
    %402 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %403 = stablehlo.multiply %cst_11, %402 : tensor<f32>
    %404 = stablehlo.broadcast_in_dim %403, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %405 = stablehlo.negate %404 : tensor<16xf32>
    %406 = stablehlo.negate %404 : tensor<16xf32>
    %407 = stablehlo.multiply %406, %233 : tensor<16xf32>
    %408 = stablehlo.divide %404, %227 : tensor<16xf32>
    %409 = stablehlo.add %405, %408 : tensor<16xf32>
    %410 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %411 = stablehlo.multiply %409, %410 : tensor<16xf32>
    %412 = stablehlo.divide %cst_15, %cst_8 : tensor<f32>
    %413 = stablehlo.broadcast_in_dim %412, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %414 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %415 = stablehlo.multiply %414, %413 : tensor<512xf32>
    %416 = stablehlo.broadcast_in_dim %415, dims = [0] : (tensor<512xf32>) -> tensor<512x16xf32>
    %417 = stablehlo.negate %416 : tensor<512x16xf32>
    %418 = stablehlo.negate %416 : tensor<512x16xf32>
    %419 = stablehlo.multiply %418, %217 : tensor<512x16xf32>
    %420 = stablehlo.divide %416, %211 : tensor<512x16xf32>
    %421 = stablehlo.add %417, %420 : tensor<512x16xf32>
    %422 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %423 = stablehlo.multiply %421, %422 : tensor<512x16xf32>
    %424 = stablehlo.divide %cst_15, %cst_15 : tensor<f32>
    %425 = stablehlo.multiply %cst_11, %424 : tensor<f32>
    %426 = stablehlo.broadcast_in_dim %425, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %427 = stablehlo.negate %426 : tensor<512xf32>
    %428 = stablehlo.negate %426 : tensor<512xf32>
    %429 = stablehlo.multiply %428, %202 : tensor<512xf32>
    %430 = stablehlo.divide %426, %196 : tensor<512xf32>
    %431 = stablehlo.add %427, %430 : tensor<512xf32>
    %432 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %433 = stablehlo.multiply %431, %432 : tensor<512xf32>
    %434 = stablehlo.divide %cst_15, %cst_9 : tensor<f32>
    %435 = stablehlo.broadcast_in_dim %434, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %436 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %437 = stablehlo.multiply %436, %435 : tensor<784xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0] : (tensor<784xf32>) -> tensor<784x512xf32>
    %439 = stablehlo.negate %438 : tensor<784x512xf32>
    %440 = stablehlo.negate %438 : tensor<784x512xf32>
    %441 = stablehlo.multiply %440, %187 : tensor<784x512xf32>
    %442 = stablehlo.divide %438, %181 : tensor<784x512xf32>
    %443 = stablehlo.add %439, %442 : tensor<784x512xf32>
    %444 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %445 = stablehlo.multiply %443, %444 : tensor<784x512xf32>
    %446 = stablehlo.divide %cst_15, %cst_12 : tensor<f32>
    %447 = stablehlo.broadcast_in_dim %446, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %448 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %449 = stablehlo.multiply %448, %447 : tensor<32x1xf32>
    %450 = stablehlo.broadcast_in_dim %449, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x16xf32>
    %451 = stablehlo.negate %450 : tensor<32x1x16xf32>
    %452 = stablehlo.multiply %451, %167 : tensor<32x1x16xf32>
    %453 = stablehlo.negate %450 : tensor<32x1x16xf32>
    %454 = stablehlo.multiply %453, %172 : tensor<32x1x16xf32>
    %455 = stablehlo.add %452, %450 : tensor<32x1x16xf32>
    %456 = stablehlo.divide %cst_15, %cst_12 : tensor<f32>
    %457 = stablehlo.broadcast_in_dim %456, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %458 = stablehlo.broadcast_in_dim %457, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x784xf32>
    %459 = stablehlo.multiply %458, %163 : tensor<32x1x784xf32>
    %460 = stablehlo.negate %459 : tensor<32x1x784xf32>
    %461 = stablehlo.multiply %460, %158 : tensor<32x1x784xf32>
    %462 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<32x1x784xf32>
    %463 = stablehlo.multiply %461, %462 : tensor<32x1x784xf32>
    %464 = stablehlo.negate %463 : tensor<32x1x784xf32>
    %465 = stablehlo.multiply %464, %151 : tensor<32x1x784xf32>
    %466 = stablehlo.negate %465 : tensor<32x1x784xf32>
    %467 = stablehlo.reduce(%466 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<32x1x784xf32>, tensor<f32>) -> tensor<32x784xf32>
    %468 = stablehlo.reduce(%467 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x784xf32>, tensor<f32>) -> tensor<784xf32>
    %469 = stablehlo.reshape %468 : (tensor<784xf32>) -> tensor<1x784xf32>
    %470 = stablehlo.reduce(%469 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784xf32>, tensor<f32>) -> tensor<784xf32>
    %471 = stablehlo.add %341, %470 : tensor<784xf32>
    %472 = stablehlo.multiply %467, %140 : tensor<32x784xf32>
    %473 = stablehlo.reduce(%472 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x784xf32>, tensor<f32>) -> tensor<784xf32>
    %474 = stablehlo.reshape %473 : (tensor<784xf32>) -> tensor<1x784xf32>
    %475 = stablehlo.reduce(%474 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784xf32>, tensor<f32>) -> tensor<784xf32>
    %476 = stablehlo.add %345, %475 : tensor<784xf32>
    %477 = stablehlo.dot_general %466, %122, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<32x1x784xf32>, tensor<32x1x512xf32>) -> tensor<32x784x512xf32>
    %478 = stablehlo.transpose %477, dims = [0, 2, 1] : (tensor<32x784x512xf32>) -> tensor<32x512x784xf32>
    %479 = stablehlo.dot_general %466, %136, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<32x1x784xf32>, tensor<32x512x784xf32>) -> tensor<32x1x512xf32>
    %480 = stablehlo.reduce(%478 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %481 = stablehlo.reshape %480 : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %482 = stablehlo.reduce(%481 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %483 = stablehlo.add %353, %482 : tensor<512x784xf32>
    %484 = stablehlo.multiply %478, %130 : tensor<32x512x784xf32>
    %485 = stablehlo.reduce(%484 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %486 = stablehlo.reshape %485 : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %487 = stablehlo.reduce(%486 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x784xf32>, tensor<f32>) -> tensor<512x784xf32>
    %488 = stablehlo.add %357, %487 : tensor<512x784xf32>
    %489 = stablehlo.multiply %479, %124 : tensor<32x1x512xf32>
    %490 = stablehlo.multiply %489, %122 : tensor<32x1x512xf32>
    %491 = stablehlo.add %489, %490 : tensor<32x1x512xf32>
    %492 = stablehlo.reduce(%491 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<32x1x512xf32>, tensor<f32>) -> tensor<32x512xf32>
    %493 = stablehlo.reduce(%492 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512xf32>, tensor<f32>) -> tensor<512xf32>
    %494 = stablehlo.reshape %493 : (tensor<512xf32>) -> tensor<1x512xf32>
    %495 = stablehlo.reduce(%494 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %496 = stablehlo.add %363, %495 : tensor<512xf32>
    %497 = stablehlo.multiply %492, %112 : tensor<32x512xf32>
    %498 = stablehlo.reduce(%497 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512xf32>, tensor<f32>) -> tensor<512xf32>
    %499 = stablehlo.reshape %498 : (tensor<512xf32>) -> tensor<1x512xf32>
    %500 = stablehlo.reduce(%499 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %501 = stablehlo.add %367, %500 : tensor<512xf32>
    %502 = stablehlo.dot_general %491, %94, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<32x1x512xf32>, tensor<32x1x16xf32>) -> tensor<32x512x16xf32>
    %503 = stablehlo.transpose %502, dims = [0, 2, 1] : (tensor<32x512x16xf32>) -> tensor<32x16x512xf32>
    %504 = stablehlo.dot_general %491, %106, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<32x1x512xf32>, tensor<32x16x512xf32>) -> tensor<32x1x16xf32>
    %505 = stablehlo.reduce(%503 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %506 = stablehlo.reshape %505 : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %507 = stablehlo.reduce(%506 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %508 = stablehlo.add %375, %507 : tensor<16x512xf32>
    %509 = stablehlo.multiply %503, %100 : tensor<32x16x512xf32>
    %510 = stablehlo.reduce(%509 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %511 = stablehlo.reshape %510 : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %512 = stablehlo.reduce(%511 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16x512xf32>, tensor<f32>) -> tensor<16x512xf32>
    %513 = stablehlo.add %379, %512 : tensor<16x512xf32>
    %514 = stablehlo.add %454, %504 : tensor<32x1x16xf32>
    %515 = stablehlo.multiply %504, %92 : tensor<32x1x16xf32>
    %516 = stablehlo.multiply %515, %86 : tensor<32x1x16xf32>
    %517 = stablehlo.add %455, %516 : tensor<32x1x16xf32>
    %518 = stablehlo.reduce(%517 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<32x1x16xf32>, tensor<f32>) -> tensor<32x16xf32>
    %519 = stablehlo.reduce(%518 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x16xf32>, tensor<f32>) -> tensor<16xf32>
    %520 = stablehlo.reshape %519 : (tensor<16xf32>) -> tensor<1x16xf32>
    %521 = stablehlo.reduce(%520 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %522 = stablehlo.add %385, %521 : tensor<16xf32>
    %523 = stablehlo.multiply %518, %76 : tensor<32x16xf32>
    %524 = stablehlo.reduce(%523 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x16xf32>, tensor<f32>) -> tensor<16xf32>
    %525 = stablehlo.reshape %524 : (tensor<16xf32>) -> tensor<1x16xf32>
    %526 = stablehlo.reduce(%525 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %527 = stablehlo.add %389, %526 : tensor<16xf32>
    %528 = stablehlo.dot_general %517, %29, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<32x1x16xf32>, tensor<32x1x512xf32>) -> tensor<32x16x512xf32>
    %529 = stablehlo.transpose %528, dims = [0, 2, 1] : (tensor<32x16x512xf32>) -> tensor<32x512x16xf32>
    %530 = stablehlo.dot_general %517, %70, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<32x1x16xf32>, tensor<32x512x16xf32>) -> tensor<32x1x512xf32>
    %531 = stablehlo.reduce(%529 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %532 = stablehlo.reshape %531 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %533 = stablehlo.reduce(%532 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %534 = stablehlo.add %397, %533 : tensor<512x16xf32>
    %535 = stablehlo.multiply %529, %64 : tensor<32x512x16xf32>
    %536 = stablehlo.reduce(%535 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %537 = stablehlo.reshape %536 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %538 = stablehlo.reduce(%537 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %539 = stablehlo.add %401, %538 : tensor<512x16xf32>
    %540 = stablehlo.reduce(%514 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<32x1x16xf32>, tensor<f32>) -> tensor<32x16xf32>
    %541 = stablehlo.reduce(%540 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x16xf32>, tensor<f32>) -> tensor<16xf32>
    %542 = stablehlo.reshape %541 : (tensor<16xf32>) -> tensor<1x16xf32>
    %543 = stablehlo.reduce(%542 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %544 = stablehlo.add %407, %543 : tensor<16xf32>
    %545 = stablehlo.multiply %540, %49 : tensor<32x16xf32>
    %546 = stablehlo.reduce(%545 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x16xf32>, tensor<f32>) -> tensor<16xf32>
    %547 = stablehlo.reshape %546 : (tensor<16xf32>) -> tensor<1x16xf32>
    %548 = stablehlo.reduce(%547 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x16xf32>, tensor<f32>) -> tensor<16xf32>
    %549 = stablehlo.add %411, %548 : tensor<16xf32>
    %550 = stablehlo.dot_general %514, %29, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<32x1x16xf32>, tensor<32x1x512xf32>) -> tensor<32x16x512xf32>
    %551 = stablehlo.transpose %550, dims = [0, 2, 1] : (tensor<32x16x512xf32>) -> tensor<32x512x16xf32>
    %552 = stablehlo.dot_general %514, %43, batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<32x1x16xf32>, tensor<32x512x16xf32>) -> tensor<32x1x512xf32>
    %553 = stablehlo.add %530, %552 : tensor<32x1x512xf32>
    %554 = stablehlo.reduce(%551 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %555 = stablehlo.reshape %554 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %556 = stablehlo.reduce(%555 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %557 = stablehlo.add %419, %556 : tensor<512x16xf32>
    %558 = stablehlo.multiply %551, %37 : tensor<32x512x16xf32>
    %559 = stablehlo.reduce(%558 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %560 = stablehlo.reshape %559 : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %561 = stablehlo.reduce(%560 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512x16xf32>, tensor<f32>) -> tensor<512x16xf32>
    %562 = stablehlo.add %423, %561 : tensor<512x16xf32>
    %563 = stablehlo.multiply %553, %31 : tensor<32x1x512xf32>
    %564 = stablehlo.multiply %563, %29 : tensor<32x1x512xf32>
    %565 = stablehlo.add %563, %564 : tensor<32x1x512xf32>
    %566 = stablehlo.reduce(%565 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<32x1x512xf32>, tensor<f32>) -> tensor<32x512xf32>
    %567 = stablehlo.reduce(%566 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512xf32>, tensor<f32>) -> tensor<512xf32>
    %568 = stablehlo.reshape %567 : (tensor<512xf32>) -> tensor<1x512xf32>
    %569 = stablehlo.reduce(%568 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %570 = stablehlo.add %429, %569 : tensor<512xf32>
    %571 = stablehlo.multiply %566, %18 : tensor<32x512xf32>
    %572 = stablehlo.reduce(%571 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x512xf32>, tensor<f32>) -> tensor<512xf32>
    %573 = stablehlo.reshape %572 : (tensor<512xf32>) -> tensor<1x512xf32>
    %574 = stablehlo.reduce(%573 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x512xf32>, tensor<f32>) -> tensor<512xf32>
    %575 = stablehlo.add %433, %574 : tensor<512xf32>
    %576 = stablehlo.dot_general %565, %25, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<32x1x512xf32>, tensor<32x1x784xf32>) -> tensor<32x512x784xf32>
    %577 = stablehlo.transpose %576, dims = [0, 2, 1] : (tensor<32x512x784xf32>) -> tensor<32x784x512xf32>
    %578 = stablehlo.reduce(%577 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %579 = stablehlo.reshape %578 : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %580 = stablehlo.reduce(%579 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %581 = stablehlo.add %441, %580 : tensor<784x512xf32>
    %582 = stablehlo.multiply %577, %6 : tensor<32x784x512xf32>
    %583 = stablehlo.reduce(%582 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<32x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %584 = stablehlo.reshape %583 : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %585 = stablehlo.reduce(%584 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<1x784x512xf32>, tensor<f32>) -> tensor<784x512xf32>
    %586 = stablehlo.add %445, %585 : tensor<784x512xf32>
    %587 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %588 = stablehlo.multiply %587, %581 : tensor<784x512xf32>
    %589 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %590 = stablehlo.multiply %589, %arg22 : tensor<784x512xf32>
    %591 = stablehlo.add %588, %590 : tensor<784x512xf32>
    %592 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %593 = stablehlo.multiply %592, %586 : tensor<784x512xf32>
    %594 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %595 = stablehlo.multiply %594, %arg23 : tensor<784x512xf32>
    %596 = stablehlo.add %593, %595 : tensor<784x512xf32>
    %597 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %598 = stablehlo.multiply %597, %570 : tensor<512xf32>
    %599 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %600 = stablehlo.multiply %599, %arg24 : tensor<512xf32>
    %601 = stablehlo.add %598, %600 : tensor<512xf32>
    %602 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %603 = stablehlo.multiply %602, %575 : tensor<512xf32>
    %604 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %605 = stablehlo.multiply %604, %arg25 : tensor<512xf32>
    %606 = stablehlo.add %603, %605 : tensor<512xf32>
    %607 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %608 = stablehlo.multiply %607, %557 : tensor<512x16xf32>
    %609 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %610 = stablehlo.multiply %609, %arg26 : tensor<512x16xf32>
    %611 = stablehlo.add %608, %610 : tensor<512x16xf32>
    %612 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %613 = stablehlo.multiply %612, %562 : tensor<512x16xf32>
    %614 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %615 = stablehlo.multiply %614, %arg27 : tensor<512x16xf32>
    %616 = stablehlo.add %613, %615 : tensor<512x16xf32>
    %617 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %618 = stablehlo.multiply %617, %544 : tensor<16xf32>
    %619 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %620 = stablehlo.multiply %619, %arg28 : tensor<16xf32>
    %621 = stablehlo.add %618, %620 : tensor<16xf32>
    %622 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %623 = stablehlo.multiply %622, %549 : tensor<16xf32>
    %624 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %625 = stablehlo.multiply %624, %arg29 : tensor<16xf32>
    %626 = stablehlo.add %623, %625 : tensor<16xf32>
    %627 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %628 = stablehlo.multiply %627, %534 : tensor<512x16xf32>
    %629 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %630 = stablehlo.multiply %629, %arg30 : tensor<512x16xf32>
    %631 = stablehlo.add %628, %630 : tensor<512x16xf32>
    %632 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %633 = stablehlo.multiply %632, %539 : tensor<512x16xf32>
    %634 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %635 = stablehlo.multiply %634, %arg31 : tensor<512x16xf32>
    %636 = stablehlo.add %633, %635 : tensor<512x16xf32>
    %637 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %638 = stablehlo.multiply %637, %522 : tensor<16xf32>
    %639 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %640 = stablehlo.multiply %639, %arg32 : tensor<16xf32>
    %641 = stablehlo.add %638, %640 : tensor<16xf32>
    %642 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %643 = stablehlo.multiply %642, %527 : tensor<16xf32>
    %644 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %645 = stablehlo.multiply %644, %arg33 : tensor<16xf32>
    %646 = stablehlo.add %643, %645 : tensor<16xf32>
    %647 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %648 = stablehlo.multiply %647, %508 : tensor<16x512xf32>
    %649 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %650 = stablehlo.multiply %649, %arg34 : tensor<16x512xf32>
    %651 = stablehlo.add %648, %650 : tensor<16x512xf32>
    %652 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %653 = stablehlo.multiply %652, %513 : tensor<16x512xf32>
    %654 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %655 = stablehlo.multiply %654, %arg35 : tensor<16x512xf32>
    %656 = stablehlo.add %653, %655 : tensor<16x512xf32>
    %657 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %658 = stablehlo.multiply %657, %496 : tensor<512xf32>
    %659 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %660 = stablehlo.multiply %659, %arg36 : tensor<512xf32>
    %661 = stablehlo.add %658, %660 : tensor<512xf32>
    %662 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %663 = stablehlo.multiply %662, %501 : tensor<512xf32>
    %664 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %665 = stablehlo.multiply %664, %arg37 : tensor<512xf32>
    %666 = stablehlo.add %663, %665 : tensor<512xf32>
    %667 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %668 = stablehlo.multiply %667, %483 : tensor<512x784xf32>
    %669 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %670 = stablehlo.multiply %669, %arg38 : tensor<512x784xf32>
    %671 = stablehlo.add %668, %670 : tensor<512x784xf32>
    %672 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %673 = stablehlo.multiply %672, %488 : tensor<512x784xf32>
    %674 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %675 = stablehlo.multiply %674, %arg39 : tensor<512x784xf32>
    %676 = stablehlo.add %673, %675 : tensor<512x784xf32>
    %677 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %678 = stablehlo.multiply %677, %471 : tensor<784xf32>
    %679 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %680 = stablehlo.multiply %679, %arg40 : tensor<784xf32>
    %681 = stablehlo.add %678, %680 : tensor<784xf32>
    %682 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %683 = stablehlo.multiply %682, %476 : tensor<784xf32>
    %684 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %685 = stablehlo.multiply %684, %arg41 : tensor<784xf32>
    %686 = stablehlo.add %683, %685 : tensor<784xf32>
    %687 = stablehlo.multiply %581, %581 : tensor<784x512xf32>
    %688 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %689 = stablehlo.multiply %688, %687 : tensor<784x512xf32>
    %690 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %691 = stablehlo.multiply %690, %arg42 : tensor<784x512xf32>
    %692 = stablehlo.add %689, %691 : tensor<784x512xf32>
    %693 = stablehlo.multiply %586, %586 : tensor<784x512xf32>
    %694 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %695 = stablehlo.multiply %694, %693 : tensor<784x512xf32>
    %696 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %697 = stablehlo.multiply %696, %arg43 : tensor<784x512xf32>
    %698 = stablehlo.add %695, %697 : tensor<784x512xf32>
    %699 = stablehlo.multiply %570, %570 : tensor<512xf32>
    %700 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %701 = stablehlo.multiply %700, %699 : tensor<512xf32>
    %702 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %703 = stablehlo.multiply %702, %arg44 : tensor<512xf32>
    %704 = stablehlo.add %701, %703 : tensor<512xf32>
    %705 = stablehlo.multiply %575, %575 : tensor<512xf32>
    %706 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %707 = stablehlo.multiply %706, %705 : tensor<512xf32>
    %708 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %709 = stablehlo.multiply %708, %arg45 : tensor<512xf32>
    %710 = stablehlo.add %707, %709 : tensor<512xf32>
    %711 = stablehlo.multiply %557, %557 : tensor<512x16xf32>
    %712 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %713 = stablehlo.multiply %712, %711 : tensor<512x16xf32>
    %714 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %715 = stablehlo.multiply %714, %arg46 : tensor<512x16xf32>
    %716 = stablehlo.add %713, %715 : tensor<512x16xf32>
    %717 = stablehlo.multiply %562, %562 : tensor<512x16xf32>
    %718 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %719 = stablehlo.multiply %718, %717 : tensor<512x16xf32>
    %720 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %721 = stablehlo.multiply %720, %arg47 : tensor<512x16xf32>
    %722 = stablehlo.add %719, %721 : tensor<512x16xf32>
    %723 = stablehlo.multiply %544, %544 : tensor<16xf32>
    %724 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %725 = stablehlo.multiply %724, %723 : tensor<16xf32>
    %726 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %727 = stablehlo.multiply %726, %arg48 : tensor<16xf32>
    %728 = stablehlo.add %725, %727 : tensor<16xf32>
    %729 = stablehlo.multiply %549, %549 : tensor<16xf32>
    %730 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %731 = stablehlo.multiply %730, %729 : tensor<16xf32>
    %732 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %733 = stablehlo.multiply %732, %arg49 : tensor<16xf32>
    %734 = stablehlo.add %731, %733 : tensor<16xf32>
    %735 = stablehlo.multiply %534, %534 : tensor<512x16xf32>
    %736 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %737 = stablehlo.multiply %736, %735 : tensor<512x16xf32>
    %738 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %739 = stablehlo.multiply %738, %arg50 : tensor<512x16xf32>
    %740 = stablehlo.add %737, %739 : tensor<512x16xf32>
    %741 = stablehlo.multiply %539, %539 : tensor<512x16xf32>
    %742 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %743 = stablehlo.multiply %742, %741 : tensor<512x16xf32>
    %744 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %745 = stablehlo.multiply %744, %arg51 : tensor<512x16xf32>
    %746 = stablehlo.add %743, %745 : tensor<512x16xf32>
    %747 = stablehlo.multiply %522, %522 : tensor<16xf32>
    %748 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %749 = stablehlo.multiply %748, %747 : tensor<16xf32>
    %750 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %751 = stablehlo.multiply %750, %arg52 : tensor<16xf32>
    %752 = stablehlo.add %749, %751 : tensor<16xf32>
    %753 = stablehlo.multiply %527, %527 : tensor<16xf32>
    %754 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %755 = stablehlo.multiply %754, %753 : tensor<16xf32>
    %756 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %757 = stablehlo.multiply %756, %arg53 : tensor<16xf32>
    %758 = stablehlo.add %755, %757 : tensor<16xf32>
    %759 = stablehlo.multiply %508, %508 : tensor<16x512xf32>
    %760 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %761 = stablehlo.multiply %760, %759 : tensor<16x512xf32>
    %762 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %763 = stablehlo.multiply %762, %arg54 : tensor<16x512xf32>
    %764 = stablehlo.add %761, %763 : tensor<16x512xf32>
    %765 = stablehlo.multiply %513, %513 : tensor<16x512xf32>
    %766 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %767 = stablehlo.multiply %766, %765 : tensor<16x512xf32>
    %768 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %769 = stablehlo.multiply %768, %arg55 : tensor<16x512xf32>
    %770 = stablehlo.add %767, %769 : tensor<16x512xf32>
    %771 = stablehlo.multiply %496, %496 : tensor<512xf32>
    %772 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %773 = stablehlo.multiply %772, %771 : tensor<512xf32>
    %774 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %775 = stablehlo.multiply %774, %arg56 : tensor<512xf32>
    %776 = stablehlo.add %773, %775 : tensor<512xf32>
    %777 = stablehlo.multiply %501, %501 : tensor<512xf32>
    %778 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %779 = stablehlo.multiply %778, %777 : tensor<512xf32>
    %780 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %781 = stablehlo.multiply %780, %arg57 : tensor<512xf32>
    %782 = stablehlo.add %779, %781 : tensor<512xf32>
    %783 = stablehlo.multiply %483, %483 : tensor<512x784xf32>
    %784 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %785 = stablehlo.multiply %784, %783 : tensor<512x784xf32>
    %786 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %787 = stablehlo.multiply %786, %arg58 : tensor<512x784xf32>
    %788 = stablehlo.add %785, %787 : tensor<512x784xf32>
    %789 = stablehlo.multiply %488, %488 : tensor<512x784xf32>
    %790 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %791 = stablehlo.multiply %790, %789 : tensor<512x784xf32>
    %792 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %793 = stablehlo.multiply %792, %arg59 : tensor<512x784xf32>
    %794 = stablehlo.add %791, %793 : tensor<512x784xf32>
    %795 = stablehlo.multiply %471, %471 : tensor<784xf32>
    %796 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %797 = stablehlo.multiply %796, %795 : tensor<784xf32>
    %798 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %799 = stablehlo.multiply %798, %arg60 : tensor<784xf32>
    %800 = stablehlo.add %797, %799 : tensor<784xf32>
    %801 = stablehlo.multiply %476, %476 : tensor<784xf32>
    %802 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %803 = stablehlo.multiply %802, %801 : tensor<784xf32>
    %804 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %805 = stablehlo.multiply %804, %arg61 : tensor<784xf32>
    %806 = stablehlo.add %803, %805 : tensor<784xf32>
    %807 = stablehlo.compare  LT, %arg21, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %808 = stablehlo.add %arg21, %c : tensor<i32>
    %809 = call @_where(%807, %808, %c_2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %810 = stablehlo.convert %809 : (tensor<i32>) -> tensor<f32>
    %811 = stablehlo.power %cst_5, %810 : tensor<f32>
    %812 = stablehlo.subtract %cst_15, %811 : tensor<f32>
    %813 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %814 = stablehlo.divide %591, %813 : tensor<784x512xf32>
    %815 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %816 = stablehlo.divide %596, %815 : tensor<784x512xf32>
    %817 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %818 = stablehlo.divide %601, %817 : tensor<512xf32>
    %819 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %820 = stablehlo.divide %606, %819 : tensor<512xf32>
    %821 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %822 = stablehlo.divide %611, %821 : tensor<512x16xf32>
    %823 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %824 = stablehlo.divide %616, %823 : tensor<512x16xf32>
    %825 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %826 = stablehlo.divide %621, %825 : tensor<16xf32>
    %827 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %828 = stablehlo.divide %626, %827 : tensor<16xf32>
    %829 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %830 = stablehlo.divide %631, %829 : tensor<512x16xf32>
    %831 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %832 = stablehlo.divide %636, %831 : tensor<512x16xf32>
    %833 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %834 = stablehlo.divide %641, %833 : tensor<16xf32>
    %835 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %836 = stablehlo.divide %646, %835 : tensor<16xf32>
    %837 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %838 = stablehlo.divide %651, %837 : tensor<16x512xf32>
    %839 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %840 = stablehlo.divide %656, %839 : tensor<16x512xf32>
    %841 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %842 = stablehlo.divide %661, %841 : tensor<512xf32>
    %843 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %844 = stablehlo.divide %666, %843 : tensor<512xf32>
    %845 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %846 = stablehlo.divide %671, %845 : tensor<512x784xf32>
    %847 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %848 = stablehlo.divide %676, %847 : tensor<512x784xf32>
    %849 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %850 = stablehlo.divide %681, %849 : tensor<784xf32>
    %851 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %852 = stablehlo.divide %686, %851 : tensor<784xf32>
    %853 = stablehlo.convert %809 : (tensor<i32>) -> tensor<f32>
    %854 = stablehlo.power %cst_3, %853 : tensor<f32>
    %855 = stablehlo.subtract %cst_15, %854 : tensor<f32>
    %856 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %857 = stablehlo.divide %692, %856 : tensor<784x512xf32>
    %858 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %859 = stablehlo.divide %698, %858 : tensor<784x512xf32>
    %860 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %861 = stablehlo.divide %704, %860 : tensor<512xf32>
    %862 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %863 = stablehlo.divide %710, %862 : tensor<512xf32>
    %864 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %865 = stablehlo.divide %716, %864 : tensor<512x16xf32>
    %866 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %867 = stablehlo.divide %722, %866 : tensor<512x16xf32>
    %868 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %869 = stablehlo.divide %728, %868 : tensor<16xf32>
    %870 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %871 = stablehlo.divide %734, %870 : tensor<16xf32>
    %872 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %873 = stablehlo.divide %740, %872 : tensor<512x16xf32>
    %874 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %875 = stablehlo.divide %746, %874 : tensor<512x16xf32>
    %876 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %877 = stablehlo.divide %752, %876 : tensor<16xf32>
    %878 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %879 = stablehlo.divide %758, %878 : tensor<16xf32>
    %880 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %881 = stablehlo.divide %764, %880 : tensor<16x512xf32>
    %882 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %883 = stablehlo.divide %770, %882 : tensor<16x512xf32>
    %884 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %885 = stablehlo.divide %776, %884 : tensor<512xf32>
    %886 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %887 = stablehlo.divide %782, %886 : tensor<512xf32>
    %888 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %889 = stablehlo.divide %788, %888 : tensor<512x784xf32>
    %890 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %891 = stablehlo.divide %794, %890 : tensor<512x784xf32>
    %892 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %893 = stablehlo.divide %800, %892 : tensor<784xf32>
    %894 = stablehlo.broadcast_in_dim %855, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %895 = stablehlo.divide %806, %894 : tensor<784xf32>
    %896 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %897 = stablehlo.add %857, %896 : tensor<784x512xf32>
    %898 = stablehlo.sqrt %897 : tensor<784x512xf32>
    %899 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %900 = stablehlo.add %898, %899 : tensor<784x512xf32>
    %901 = stablehlo.divide %814, %900 : tensor<784x512xf32>
    %902 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %903 = stablehlo.add %859, %902 : tensor<784x512xf32>
    %904 = stablehlo.sqrt %903 : tensor<784x512xf32>
    %905 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %906 = stablehlo.add %904, %905 : tensor<784x512xf32>
    %907 = stablehlo.divide %816, %906 : tensor<784x512xf32>
    %908 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %909 = stablehlo.add %861, %908 : tensor<512xf32>
    %910 = stablehlo.sqrt %909 : tensor<512xf32>
    %911 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %912 = stablehlo.add %910, %911 : tensor<512xf32>
    %913 = stablehlo.divide %818, %912 : tensor<512xf32>
    %914 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %915 = stablehlo.add %863, %914 : tensor<512xf32>
    %916 = stablehlo.sqrt %915 : tensor<512xf32>
    %917 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %918 = stablehlo.add %916, %917 : tensor<512xf32>
    %919 = stablehlo.divide %820, %918 : tensor<512xf32>
    %920 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %921 = stablehlo.add %865, %920 : tensor<512x16xf32>
    %922 = stablehlo.sqrt %921 : tensor<512x16xf32>
    %923 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %924 = stablehlo.add %922, %923 : tensor<512x16xf32>
    %925 = stablehlo.divide %822, %924 : tensor<512x16xf32>
    %926 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %927 = stablehlo.add %867, %926 : tensor<512x16xf32>
    %928 = stablehlo.sqrt %927 : tensor<512x16xf32>
    %929 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %930 = stablehlo.add %928, %929 : tensor<512x16xf32>
    %931 = stablehlo.divide %824, %930 : tensor<512x16xf32>
    %932 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %933 = stablehlo.add %869, %932 : tensor<16xf32>
    %934 = stablehlo.sqrt %933 : tensor<16xf32>
    %935 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %936 = stablehlo.add %934, %935 : tensor<16xf32>
    %937 = stablehlo.divide %826, %936 : tensor<16xf32>
    %938 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %939 = stablehlo.add %871, %938 : tensor<16xf32>
    %940 = stablehlo.sqrt %939 : tensor<16xf32>
    %941 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %942 = stablehlo.add %940, %941 : tensor<16xf32>
    %943 = stablehlo.divide %828, %942 : tensor<16xf32>
    %944 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %945 = stablehlo.add %873, %944 : tensor<512x16xf32>
    %946 = stablehlo.sqrt %945 : tensor<512x16xf32>
    %947 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %948 = stablehlo.add %946, %947 : tensor<512x16xf32>
    %949 = stablehlo.divide %830, %948 : tensor<512x16xf32>
    %950 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %951 = stablehlo.add %875, %950 : tensor<512x16xf32>
    %952 = stablehlo.sqrt %951 : tensor<512x16xf32>
    %953 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %954 = stablehlo.add %952, %953 : tensor<512x16xf32>
    %955 = stablehlo.divide %832, %954 : tensor<512x16xf32>
    %956 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %957 = stablehlo.add %877, %956 : tensor<16xf32>
    %958 = stablehlo.sqrt %957 : tensor<16xf32>
    %959 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %960 = stablehlo.add %958, %959 : tensor<16xf32>
    %961 = stablehlo.divide %834, %960 : tensor<16xf32>
    %962 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %963 = stablehlo.add %879, %962 : tensor<16xf32>
    %964 = stablehlo.sqrt %963 : tensor<16xf32>
    %965 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %966 = stablehlo.add %964, %965 : tensor<16xf32>
    %967 = stablehlo.divide %836, %966 : tensor<16xf32>
    %968 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %969 = stablehlo.add %881, %968 : tensor<16x512xf32>
    %970 = stablehlo.sqrt %969 : tensor<16x512xf32>
    %971 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %972 = stablehlo.add %970, %971 : tensor<16x512xf32>
    %973 = stablehlo.divide %838, %972 : tensor<16x512xf32>
    %974 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %975 = stablehlo.add %883, %974 : tensor<16x512xf32>
    %976 = stablehlo.sqrt %975 : tensor<16x512xf32>
    %977 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %978 = stablehlo.add %976, %977 : tensor<16x512xf32>
    %979 = stablehlo.divide %840, %978 : tensor<16x512xf32>
    %980 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %981 = stablehlo.add %885, %980 : tensor<512xf32>
    %982 = stablehlo.sqrt %981 : tensor<512xf32>
    %983 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %984 = stablehlo.add %982, %983 : tensor<512xf32>
    %985 = stablehlo.divide %842, %984 : tensor<512xf32>
    %986 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %987 = stablehlo.add %887, %986 : tensor<512xf32>
    %988 = stablehlo.sqrt %987 : tensor<512xf32>
    %989 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %990 = stablehlo.add %988, %989 : tensor<512xf32>
    %991 = stablehlo.divide %844, %990 : tensor<512xf32>
    %992 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %993 = stablehlo.add %889, %992 : tensor<512x784xf32>
    %994 = stablehlo.sqrt %993 : tensor<512x784xf32>
    %995 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %996 = stablehlo.add %994, %995 : tensor<512x784xf32>
    %997 = stablehlo.divide %846, %996 : tensor<512x784xf32>
    %998 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %999 = stablehlo.add %891, %998 : tensor<512x784xf32>
    %1000 = stablehlo.sqrt %999 : tensor<512x784xf32>
    %1001 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1002 = stablehlo.add %1000, %1001 : tensor<512x784xf32>
    %1003 = stablehlo.divide %848, %1002 : tensor<512x784xf32>
    %1004 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1005 = stablehlo.add %893, %1004 : tensor<784xf32>
    %1006 = stablehlo.sqrt %1005 : tensor<784xf32>
    %1007 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1008 = stablehlo.add %1006, %1007 : tensor<784xf32>
    %1009 = stablehlo.divide %850, %1008 : tensor<784xf32>
    %1010 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1011 = stablehlo.add %895, %1010 : tensor<784xf32>
    %1012 = stablehlo.sqrt %1011 : tensor<784xf32>
    %1013 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1014 = stablehlo.add %1012, %1013 : tensor<784xf32>
    %1015 = stablehlo.divide %852, %1014 : tensor<784xf32>
    %1016 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %1017 = stablehlo.multiply %1016, %arg1 : tensor<784x512xf32>
    %1018 = stablehlo.add %901, %1017 : tensor<784x512xf32>
    %1019 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %1020 = stablehlo.multiply %1019, %arg2 : tensor<784x512xf32>
    %1021 = stablehlo.add %907, %1020 : tensor<784x512xf32>
    %1022 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1023 = stablehlo.multiply %1022, %arg3 : tensor<512xf32>
    %1024 = stablehlo.add %913, %1023 : tensor<512xf32>
    %1025 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1026 = stablehlo.multiply %1025, %arg4 : tensor<512xf32>
    %1027 = stablehlo.add %919, %1026 : tensor<512xf32>
    %1028 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1029 = stablehlo.multiply %1028, %arg5 : tensor<512x16xf32>
    %1030 = stablehlo.add %925, %1029 : tensor<512x16xf32>
    %1031 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1032 = stablehlo.multiply %1031, %arg6 : tensor<512x16xf32>
    %1033 = stablehlo.add %931, %1032 : tensor<512x16xf32>
    %1034 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1035 = stablehlo.multiply %1034, %arg7 : tensor<16xf32>
    %1036 = stablehlo.add %937, %1035 : tensor<16xf32>
    %1037 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1038 = stablehlo.multiply %1037, %arg8 : tensor<16xf32>
    %1039 = stablehlo.add %943, %1038 : tensor<16xf32>
    %1040 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1041 = stablehlo.multiply %1040, %arg9 : tensor<512x16xf32>
    %1042 = stablehlo.add %949, %1041 : tensor<512x16xf32>
    %1043 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1044 = stablehlo.multiply %1043, %arg10 : tensor<512x16xf32>
    %1045 = stablehlo.add %955, %1044 : tensor<512x16xf32>
    %1046 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1047 = stablehlo.multiply %1046, %arg11 : tensor<16xf32>
    %1048 = stablehlo.add %961, %1047 : tensor<16xf32>
    %1049 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1050 = stablehlo.multiply %1049, %arg12 : tensor<16xf32>
    %1051 = stablehlo.add %967, %1050 : tensor<16xf32>
    %1052 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1053 = stablehlo.multiply %1052, %arg13 : tensor<16x512xf32>
    %1054 = stablehlo.add %973, %1053 : tensor<16x512xf32>
    %1055 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1056 = stablehlo.multiply %1055, %arg14 : tensor<16x512xf32>
    %1057 = stablehlo.add %979, %1056 : tensor<16x512xf32>
    %1058 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1059 = stablehlo.multiply %1058, %arg15 : tensor<512xf32>
    %1060 = stablehlo.add %985, %1059 : tensor<512xf32>
    %1061 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1062 = stablehlo.multiply %1061, %arg16 : tensor<512xf32>
    %1063 = stablehlo.add %991, %1062 : tensor<512xf32>
    %1064 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1065 = stablehlo.multiply %1064, %arg17 : tensor<512x784xf32>
    %1066 = stablehlo.add %997, %1065 : tensor<512x784xf32>
    %1067 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1068 = stablehlo.multiply %1067, %arg18 : tensor<512x784xf32>
    %1069 = stablehlo.add %1003, %1068 : tensor<512x784xf32>
    %1070 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1071 = stablehlo.multiply %1070, %arg19 : tensor<784xf32>
    %1072 = stablehlo.add %1009, %1071 : tensor<784xf32>
    %1073 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1074 = stablehlo.multiply %1073, %arg20 : tensor<784xf32>
    %1075 = stablehlo.add %1015, %1074 : tensor<784xf32>
    %1076 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %1077 = stablehlo.multiply %1076, %1018 : tensor<784x512xf32>
    %1078 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %1079 = stablehlo.multiply %1078, %1021 : tensor<784x512xf32>
    %1080 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1081 = stablehlo.multiply %1080, %1024 : tensor<512xf32>
    %1082 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1083 = stablehlo.multiply %1082, %1027 : tensor<512xf32>
    %1084 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1085 = stablehlo.multiply %1084, %1030 : tensor<512x16xf32>
    %1086 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1087 = stablehlo.multiply %1086, %1033 : tensor<512x16xf32>
    %1088 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1089 = stablehlo.multiply %1088, %1036 : tensor<16xf32>
    %1090 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1091 = stablehlo.multiply %1090, %1039 : tensor<16xf32>
    %1092 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1093 = stablehlo.multiply %1092, %1042 : tensor<512x16xf32>
    %1094 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %1095 = stablehlo.multiply %1094, %1045 : tensor<512x16xf32>
    %1096 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1097 = stablehlo.multiply %1096, %1048 : tensor<16xf32>
    %1098 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %1099 = stablehlo.multiply %1098, %1051 : tensor<16xf32>
    %1100 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1101 = stablehlo.multiply %1100, %1054 : tensor<16x512xf32>
    %1102 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %1103 = stablehlo.multiply %1102, %1057 : tensor<16x512xf32>
    %1104 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1105 = stablehlo.multiply %1104, %1060 : tensor<512xf32>
    %1106 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %1107 = stablehlo.multiply %1106, %1063 : tensor<512xf32>
    %1108 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1109 = stablehlo.multiply %1108, %1066 : tensor<512x784xf32>
    %1110 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %1111 = stablehlo.multiply %1110, %1069 : tensor<512x784xf32>
    %1112 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1113 = stablehlo.multiply %1112, %1072 : tensor<784xf32>
    %1114 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %1115 = stablehlo.multiply %1114, %1075 : tensor<784xf32>
    %1116 = stablehlo.add %arg1, %1077 : tensor<784x512xf32>
    %1117 = stablehlo.add %arg2, %1079 : tensor<784x512xf32>
    %1118 = stablehlo.add %arg3, %1081 : tensor<512xf32>
    %1119 = stablehlo.add %arg4, %1083 : tensor<512xf32>
    %1120 = stablehlo.add %arg5, %1085 : tensor<512x16xf32>
    %1121 = stablehlo.add %arg6, %1087 : tensor<512x16xf32>
    %1122 = stablehlo.add %arg7, %1089 : tensor<16xf32>
    %1123 = stablehlo.add %arg8, %1091 : tensor<16xf32>
    %1124 = stablehlo.add %arg9, %1093 : tensor<512x16xf32>
    %1125 = stablehlo.add %arg10, %1095 : tensor<512x16xf32>
    %1126 = stablehlo.add %arg11, %1097 : tensor<16xf32>
    %1127 = stablehlo.add %arg12, %1099 : tensor<16xf32>
    %1128 = stablehlo.add %arg13, %1101 : tensor<16x512xf32>
    %1129 = stablehlo.add %arg14, %1103 : tensor<16x512xf32>
    %1130 = stablehlo.add %arg15, %1105 : tensor<512xf32>
    %1131 = stablehlo.add %arg16, %1107 : tensor<512xf32>
    %1132 = stablehlo.add %arg17, %1109 : tensor<512x784xf32>
    %1133 = stablehlo.add %arg18, %1111 : tensor<512x784xf32>
    %1134 = stablehlo.add %arg19, %1113 : tensor<784xf32>
    %1135 = stablehlo.add %arg20, %1115 : tensor<784xf32>
    return %1116, %1117, %1118, %1119, %1120, %1121, %1122, %1123, %1124, %1125, %1126, %1127, %1128, %1129, %1130, %1131, %1132, %1133, %1134, %1135, %809, %591, %596, %601, %606, %611, %616, %621, %626, %631, %636, %641, %646, %651, %656, %661, %666, %671, %676, %681, %686, %692, %698, %704, %710, %716, %722, %728, %734, %740, %746, %752, %758, %764, %770, %776, %782, %788, %794, %800, %806, %335 : tensor<784x512xf32>, tensor<784x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x512xf32>, tensor<16x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x784xf32>, tensor<512x784xf32>, tensor<784xf32>, tensor<784xf32>, tensor<i32>, tensor<784x512xf32>, tensor<784x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x512xf32>, tensor<16x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x784xf32>, tensor<512x784xf32>, tensor<784xf32>, tensor<784xf32>, tensor<784x512xf32>, tensor<784x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<512x16xf32>, tensor<512x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x512xf32>, tensor<16x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x784xf32>, tensor<512x784xf32>, tensor<784xf32>, tensor<784xf32>, tensor<f32>
  }
  func.func private @_threefry_split(%arg0: tensor<2xui32>) -> tensor<2x2xui32> {
    %0 = stablehlo.iota dim = 0 : tensor<4xui32>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %0 [0:2] : (tensor<4xui32>) -> tensor<2xui32>
    %6 = stablehlo.slice %0 [2:4] : (tensor<4xui32>) -> tensor<2xui32>
    %7:2 = call @threefry2x32(%2, %4, %5, %6) : (tensor<ui32>, tensor<ui32>, tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>, tensor<2xui32>)
    %8 = stablehlo.concatenate %7#0, %7#1, dim = 0 : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %9 = stablehlo.reshape %8 : (tensor<4xui32>) -> tensor<2x2xui32>
    return %9 : tensor<2x2xui32>
  }
  func.func private @threefry2x32(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<2xui32>, %arg3: tensor<2xui32>) -> (tensor<2xui32>, tensor<2xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<2xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<2xui32>
    %6 = stablehlo.add %3, %5 : tensor<2xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<2xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<2xui32>
    %11 = stablehlo.or %8, %10 : tensor<2xui32>
    %12 = stablehlo.xor %6, %11 : tensor<2xui32>
    %13 = stablehlo.add %6, %12 : tensor<2xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<2xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<2xui32>
    %18 = stablehlo.or %15, %17 : tensor<2xui32>
    %19 = stablehlo.xor %13, %18 : tensor<2xui32>
    %20 = stablehlo.add %13, %19 : tensor<2xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<2xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<2xui32>
    %25 = stablehlo.or %22, %24 : tensor<2xui32>
    %26 = stablehlo.xor %20, %25 : tensor<2xui32>
    %27 = stablehlo.add %20, %26 : tensor<2xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<2xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<2xui32>
    %32 = stablehlo.or %29, %31 : tensor<2xui32>
    %33 = stablehlo.xor %27, %32 : tensor<2xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %35 = stablehlo.add %27, %34 : tensor<2xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %37 = stablehlo.add %33, %36 : tensor<2xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %39 = stablehlo.add %37, %38 : tensor<2xui32>
    %40 = stablehlo.add %35, %39 : tensor<2xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<2xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<2xui32>
    %45 = stablehlo.or %42, %44 : tensor<2xui32>
    %46 = stablehlo.xor %40, %45 : tensor<2xui32>
    %47 = stablehlo.add %40, %46 : tensor<2xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<2xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<2xui32>
    %52 = stablehlo.or %49, %51 : tensor<2xui32>
    %53 = stablehlo.xor %47, %52 : tensor<2xui32>
    %54 = stablehlo.add %47, %53 : tensor<2xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<2xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<2xui32>
    %59 = stablehlo.or %56, %58 : tensor<2xui32>
    %60 = stablehlo.xor %54, %59 : tensor<2xui32>
    %61 = stablehlo.add %54, %60 : tensor<2xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<2xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<2xui32>
    %66 = stablehlo.or %63, %65 : tensor<2xui32>
    %67 = stablehlo.xor %61, %66 : tensor<2xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %69 = stablehlo.add %61, %68 : tensor<2xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %71 = stablehlo.add %67, %70 : tensor<2xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %73 = stablehlo.add %71, %72 : tensor<2xui32>
    %74 = stablehlo.add %69, %73 : tensor<2xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<2xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<2xui32>
    %79 = stablehlo.or %76, %78 : tensor<2xui32>
    %80 = stablehlo.xor %74, %79 : tensor<2xui32>
    %81 = stablehlo.add %74, %80 : tensor<2xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<2xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<2xui32>
    %86 = stablehlo.or %83, %85 : tensor<2xui32>
    %87 = stablehlo.xor %81, %86 : tensor<2xui32>
    %88 = stablehlo.add %81, %87 : tensor<2xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<2xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<2xui32>
    %93 = stablehlo.or %90, %92 : tensor<2xui32>
    %94 = stablehlo.xor %88, %93 : tensor<2xui32>
    %95 = stablehlo.add %88, %94 : tensor<2xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<2xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<2xui32>
    %100 = stablehlo.or %97, %99 : tensor<2xui32>
    %101 = stablehlo.xor %95, %100 : tensor<2xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %103 = stablehlo.add %95, %102 : tensor<2xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %105 = stablehlo.add %101, %104 : tensor<2xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %107 = stablehlo.add %105, %106 : tensor<2xui32>
    %108 = stablehlo.add %103, %107 : tensor<2xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<2xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<2xui32>
    %113 = stablehlo.or %110, %112 : tensor<2xui32>
    %114 = stablehlo.xor %108, %113 : tensor<2xui32>
    %115 = stablehlo.add %108, %114 : tensor<2xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<2xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<2xui32>
    %120 = stablehlo.or %117, %119 : tensor<2xui32>
    %121 = stablehlo.xor %115, %120 : tensor<2xui32>
    %122 = stablehlo.add %115, %121 : tensor<2xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<2xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<2xui32>
    %127 = stablehlo.or %124, %126 : tensor<2xui32>
    %128 = stablehlo.xor %122, %127 : tensor<2xui32>
    %129 = stablehlo.add %122, %128 : tensor<2xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<2xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<2xui32>
    %134 = stablehlo.or %131, %133 : tensor<2xui32>
    %135 = stablehlo.xor %129, %134 : tensor<2xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %137 = stablehlo.add %129, %136 : tensor<2xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %139 = stablehlo.add %135, %138 : tensor<2xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %141 = stablehlo.add %139, %140 : tensor<2xui32>
    %142 = stablehlo.add %137, %141 : tensor<2xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<2xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<2xui32>
    %147 = stablehlo.or %144, %146 : tensor<2xui32>
    %148 = stablehlo.xor %142, %147 : tensor<2xui32>
    %149 = stablehlo.add %142, %148 : tensor<2xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<2xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<2xui32>
    %154 = stablehlo.or %151, %153 : tensor<2xui32>
    %155 = stablehlo.xor %149, %154 : tensor<2xui32>
    %156 = stablehlo.add %149, %155 : tensor<2xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<2xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<2xui32>
    %161 = stablehlo.or %158, %160 : tensor<2xui32>
    %162 = stablehlo.xor %156, %161 : tensor<2xui32>
    %163 = stablehlo.add %156, %162 : tensor<2xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<2xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<2xui32>
    %168 = stablehlo.or %165, %167 : tensor<2xui32>
    %169 = stablehlo.xor %163, %168 : tensor<2xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %171 = stablehlo.add %163, %170 : tensor<2xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %173 = stablehlo.add %169, %172 : tensor<2xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<2xui32>
    %175 = stablehlo.add %173, %174 : tensor<2xui32>
    return %171, %175 : tensor<2xui32>, tensor<2xui32>
  }
  func.func private @_normal(%arg0: tensor<2xui32>) -> tensor<32x784x512xf32> {
    %0 = call @_normal_real(%arg0) : (tensor<2xui32>) -> tensor<32x784x512xf32>
    return %0 : tensor<32x784x512xf32>
  }
  func.func private @_normal_real(%arg0: tensor<2xui32>) -> tensor<32x784x512xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x784x512xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x784x512xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x784x512xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x784x512xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x784x512xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x784x512xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x784x512xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x784x512xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x784x512xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x784x512xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x784x512xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x784x512xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x784x512xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x784x512xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x784x512xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x784x512xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x784x512xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x784x512xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x784x512xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x784x512xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x784x512xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x784x512xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x784x512xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x784x512xf32>
    %1 = stablehlo.negate %0 : tensor<32x784x512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x784x512xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x784x512xf32>
    %4 = stablehlo.negate %3 : tensor<32x784x512xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x784x512xf32>, tensor<32x784x512xf32>) -> tensor<32x784x512xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x784x512xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x784x512xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x784x512xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x784x512xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x784x512xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x784x512xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x784x512xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x784x512xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x784x512xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x784x512xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x784x512xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x784x512xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x784x512xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x784x512xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x784x512xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x784x512xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x784x512xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x784x512xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x784x512xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x784x512xf32>
    %36 = stablehlo.abs %0 : tensor<32x784x512xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x784x512xf32>, tensor<32x784x512xf32>) -> tensor<32x784x512xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x784x512xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x784x512xi1>, tensor<32x784x512xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x784x512xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x784x512xf32>
    return %41 : tensor<32x784x512xf32>
  }
  func.func private @_uniform(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x784x512xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<12845056xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:6422528] : (tensor<12845056xui32>) -> tensor<6422528xui32>
    %8 = stablehlo.slice %2 [6422528:12845056] : (tensor<12845056xui32>) -> tensor<6422528xui32>
    %9:2 = call @threefry2x32_0(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<6422528xui32>, tensor<6422528xui32>) -> (tensor<6422528xui32>, tensor<6422528xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<6422528xui32>, tensor<6422528xui32>) -> tensor<12845056xui32>
    %11 = stablehlo.reshape %10 : (tensor<12845056xui32>) -> tensor<32x784x512xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x784x512xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x784x512xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x784x512xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x784x512xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x784x512xui32>) -> tensor<32x784x512xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x784x512xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x784x512xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x784x512xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x784x512xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x784x512xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x784x512xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x784x512xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x784x512xf32>
    return %25 : tensor<32x784x512xf32>
  }
  func.func private @threefry2x32_0(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<6422528xui32>, %arg3: tensor<6422528xui32>) -> (tensor<6422528xui32>, tensor<6422528xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<6422528xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<6422528xui32>
    %6 = stablehlo.add %3, %5 : tensor<6422528xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<6422528xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<6422528xui32>
    %11 = stablehlo.or %8, %10 : tensor<6422528xui32>
    %12 = stablehlo.xor %6, %11 : tensor<6422528xui32>
    %13 = stablehlo.add %6, %12 : tensor<6422528xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<6422528xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<6422528xui32>
    %18 = stablehlo.or %15, %17 : tensor<6422528xui32>
    %19 = stablehlo.xor %13, %18 : tensor<6422528xui32>
    %20 = stablehlo.add %13, %19 : tensor<6422528xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<6422528xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<6422528xui32>
    %25 = stablehlo.or %22, %24 : tensor<6422528xui32>
    %26 = stablehlo.xor %20, %25 : tensor<6422528xui32>
    %27 = stablehlo.add %20, %26 : tensor<6422528xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<6422528xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<6422528xui32>
    %32 = stablehlo.or %29, %31 : tensor<6422528xui32>
    %33 = stablehlo.xor %27, %32 : tensor<6422528xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %35 = stablehlo.add %27, %34 : tensor<6422528xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %37 = stablehlo.add %33, %36 : tensor<6422528xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %39 = stablehlo.add %37, %38 : tensor<6422528xui32>
    %40 = stablehlo.add %35, %39 : tensor<6422528xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<6422528xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<6422528xui32>
    %45 = stablehlo.or %42, %44 : tensor<6422528xui32>
    %46 = stablehlo.xor %40, %45 : tensor<6422528xui32>
    %47 = stablehlo.add %40, %46 : tensor<6422528xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<6422528xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<6422528xui32>
    %52 = stablehlo.or %49, %51 : tensor<6422528xui32>
    %53 = stablehlo.xor %47, %52 : tensor<6422528xui32>
    %54 = stablehlo.add %47, %53 : tensor<6422528xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<6422528xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<6422528xui32>
    %59 = stablehlo.or %56, %58 : tensor<6422528xui32>
    %60 = stablehlo.xor %54, %59 : tensor<6422528xui32>
    %61 = stablehlo.add %54, %60 : tensor<6422528xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<6422528xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<6422528xui32>
    %66 = stablehlo.or %63, %65 : tensor<6422528xui32>
    %67 = stablehlo.xor %61, %66 : tensor<6422528xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %69 = stablehlo.add %61, %68 : tensor<6422528xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %71 = stablehlo.add %67, %70 : tensor<6422528xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %73 = stablehlo.add %71, %72 : tensor<6422528xui32>
    %74 = stablehlo.add %69, %73 : tensor<6422528xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<6422528xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<6422528xui32>
    %79 = stablehlo.or %76, %78 : tensor<6422528xui32>
    %80 = stablehlo.xor %74, %79 : tensor<6422528xui32>
    %81 = stablehlo.add %74, %80 : tensor<6422528xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<6422528xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<6422528xui32>
    %86 = stablehlo.or %83, %85 : tensor<6422528xui32>
    %87 = stablehlo.xor %81, %86 : tensor<6422528xui32>
    %88 = stablehlo.add %81, %87 : tensor<6422528xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<6422528xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<6422528xui32>
    %93 = stablehlo.or %90, %92 : tensor<6422528xui32>
    %94 = stablehlo.xor %88, %93 : tensor<6422528xui32>
    %95 = stablehlo.add %88, %94 : tensor<6422528xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<6422528xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<6422528xui32>
    %100 = stablehlo.or %97, %99 : tensor<6422528xui32>
    %101 = stablehlo.xor %95, %100 : tensor<6422528xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %103 = stablehlo.add %95, %102 : tensor<6422528xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %105 = stablehlo.add %101, %104 : tensor<6422528xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %107 = stablehlo.add %105, %106 : tensor<6422528xui32>
    %108 = stablehlo.add %103, %107 : tensor<6422528xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<6422528xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<6422528xui32>
    %113 = stablehlo.or %110, %112 : tensor<6422528xui32>
    %114 = stablehlo.xor %108, %113 : tensor<6422528xui32>
    %115 = stablehlo.add %108, %114 : tensor<6422528xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<6422528xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<6422528xui32>
    %120 = stablehlo.or %117, %119 : tensor<6422528xui32>
    %121 = stablehlo.xor %115, %120 : tensor<6422528xui32>
    %122 = stablehlo.add %115, %121 : tensor<6422528xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<6422528xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<6422528xui32>
    %127 = stablehlo.or %124, %126 : tensor<6422528xui32>
    %128 = stablehlo.xor %122, %127 : tensor<6422528xui32>
    %129 = stablehlo.add %122, %128 : tensor<6422528xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<6422528xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<6422528xui32>
    %134 = stablehlo.or %131, %133 : tensor<6422528xui32>
    %135 = stablehlo.xor %129, %134 : tensor<6422528xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %137 = stablehlo.add %129, %136 : tensor<6422528xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %139 = stablehlo.add %135, %138 : tensor<6422528xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %141 = stablehlo.add %139, %140 : tensor<6422528xui32>
    %142 = stablehlo.add %137, %141 : tensor<6422528xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<6422528xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<6422528xui32>
    %147 = stablehlo.or %144, %146 : tensor<6422528xui32>
    %148 = stablehlo.xor %142, %147 : tensor<6422528xui32>
    %149 = stablehlo.add %142, %148 : tensor<6422528xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<6422528xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<6422528xui32>
    %154 = stablehlo.or %151, %153 : tensor<6422528xui32>
    %155 = stablehlo.xor %149, %154 : tensor<6422528xui32>
    %156 = stablehlo.add %149, %155 : tensor<6422528xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<6422528xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<6422528xui32>
    %161 = stablehlo.or %158, %160 : tensor<6422528xui32>
    %162 = stablehlo.xor %156, %161 : tensor<6422528xui32>
    %163 = stablehlo.add %156, %162 : tensor<6422528xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<6422528xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<6422528xui32>
    %168 = stablehlo.or %165, %167 : tensor<6422528xui32>
    %169 = stablehlo.xor %163, %168 : tensor<6422528xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %171 = stablehlo.add %163, %170 : tensor<6422528xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %173 = stablehlo.add %169, %172 : tensor<6422528xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<6422528xui32>
    %175 = stablehlo.add %173, %174 : tensor<6422528xui32>
    return %171, %175 : tensor<6422528xui32>, tensor<6422528xui32>
  }
  func.func private @_normal_1(%arg0: tensor<2xui32>) -> tensor<32x512xf32> {
    %0 = call @_normal_real_2(%arg0) : (tensor<2xui32>) -> tensor<32x512xf32>
    return %0 : tensor<32x512xf32>
  }
  func.func private @_normal_real_2(%arg0: tensor<2xui32>) -> tensor<32x512xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x512xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x512xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x512xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x512xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x512xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x512xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x512xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x512xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x512xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x512xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x512xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x512xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x512xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x512xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x512xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x512xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x512xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x512xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x512xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x512xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x512xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x512xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x512xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_3(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x512xf32>
    %1 = stablehlo.negate %0 : tensor<32x512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x512xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x512xf32>
    %4 = stablehlo.negate %3 : tensor<32x512xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x512xf32>, tensor<32x512xf32>) -> tensor<32x512xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x512xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x512xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x512xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x512xi1>, tensor<32x512xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x512xi1>, tensor<32x512xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x512xi1>, tensor<32x512xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x512xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x512xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x512xi1>, tensor<32x512xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x512xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x512xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x512xi1>, tensor<32x512xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x512xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x512xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x512xi1>, tensor<32x512xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x512xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x512xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x512xi1>, tensor<32x512xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x512xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x512xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x512xi1>, tensor<32x512xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x512xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x512xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x512xi1>, tensor<32x512xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x512xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x512xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x512xi1>, tensor<32x512xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x512xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x512xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x512xf32>
    %36 = stablehlo.abs %0 : tensor<32x512xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x512xf32>, tensor<32x512xf32>) -> tensor<32x512xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x512xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x512xi1>, tensor<32x512xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x512xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x512xf32>
    return %41 : tensor<32x512xf32>
  }
  func.func private @_uniform_3(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x512xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<16384xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:8192] : (tensor<16384xui32>) -> tensor<8192xui32>
    %8 = stablehlo.slice %2 [8192:16384] : (tensor<16384xui32>) -> tensor<8192xui32>
    %9:2 = call @threefry2x32_4(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<8192xui32>, tensor<8192xui32>) -> (tensor<8192xui32>, tensor<8192xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<8192xui32>, tensor<8192xui32>) -> tensor<16384xui32>
    %11 = stablehlo.reshape %10 : (tensor<16384xui32>) -> tensor<32x512xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x512xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x512xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x512xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x512xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x512xui32>) -> tensor<32x512xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x512xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x512xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x512xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x512xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x512xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x512xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x512xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x512xf32>
    return %25 : tensor<32x512xf32>
  }
  func.func private @threefry2x32_4(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<8192xui32>, %arg3: tensor<8192xui32>) -> (tensor<8192xui32>, tensor<8192xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<8192xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<8192xui32>
    %6 = stablehlo.add %3, %5 : tensor<8192xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<8192xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<8192xui32>
    %11 = stablehlo.or %8, %10 : tensor<8192xui32>
    %12 = stablehlo.xor %6, %11 : tensor<8192xui32>
    %13 = stablehlo.add %6, %12 : tensor<8192xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<8192xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<8192xui32>
    %18 = stablehlo.or %15, %17 : tensor<8192xui32>
    %19 = stablehlo.xor %13, %18 : tensor<8192xui32>
    %20 = stablehlo.add %13, %19 : tensor<8192xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<8192xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<8192xui32>
    %25 = stablehlo.or %22, %24 : tensor<8192xui32>
    %26 = stablehlo.xor %20, %25 : tensor<8192xui32>
    %27 = stablehlo.add %20, %26 : tensor<8192xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<8192xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<8192xui32>
    %32 = stablehlo.or %29, %31 : tensor<8192xui32>
    %33 = stablehlo.xor %27, %32 : tensor<8192xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %35 = stablehlo.add %27, %34 : tensor<8192xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %37 = stablehlo.add %33, %36 : tensor<8192xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %39 = stablehlo.add %37, %38 : tensor<8192xui32>
    %40 = stablehlo.add %35, %39 : tensor<8192xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<8192xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<8192xui32>
    %45 = stablehlo.or %42, %44 : tensor<8192xui32>
    %46 = stablehlo.xor %40, %45 : tensor<8192xui32>
    %47 = stablehlo.add %40, %46 : tensor<8192xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<8192xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<8192xui32>
    %52 = stablehlo.or %49, %51 : tensor<8192xui32>
    %53 = stablehlo.xor %47, %52 : tensor<8192xui32>
    %54 = stablehlo.add %47, %53 : tensor<8192xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<8192xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<8192xui32>
    %59 = stablehlo.or %56, %58 : tensor<8192xui32>
    %60 = stablehlo.xor %54, %59 : tensor<8192xui32>
    %61 = stablehlo.add %54, %60 : tensor<8192xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<8192xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<8192xui32>
    %66 = stablehlo.or %63, %65 : tensor<8192xui32>
    %67 = stablehlo.xor %61, %66 : tensor<8192xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %69 = stablehlo.add %61, %68 : tensor<8192xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %71 = stablehlo.add %67, %70 : tensor<8192xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %73 = stablehlo.add %71, %72 : tensor<8192xui32>
    %74 = stablehlo.add %69, %73 : tensor<8192xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<8192xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<8192xui32>
    %79 = stablehlo.or %76, %78 : tensor<8192xui32>
    %80 = stablehlo.xor %74, %79 : tensor<8192xui32>
    %81 = stablehlo.add %74, %80 : tensor<8192xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<8192xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<8192xui32>
    %86 = stablehlo.or %83, %85 : tensor<8192xui32>
    %87 = stablehlo.xor %81, %86 : tensor<8192xui32>
    %88 = stablehlo.add %81, %87 : tensor<8192xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<8192xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<8192xui32>
    %93 = stablehlo.or %90, %92 : tensor<8192xui32>
    %94 = stablehlo.xor %88, %93 : tensor<8192xui32>
    %95 = stablehlo.add %88, %94 : tensor<8192xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<8192xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<8192xui32>
    %100 = stablehlo.or %97, %99 : tensor<8192xui32>
    %101 = stablehlo.xor %95, %100 : tensor<8192xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %103 = stablehlo.add %95, %102 : tensor<8192xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %105 = stablehlo.add %101, %104 : tensor<8192xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %107 = stablehlo.add %105, %106 : tensor<8192xui32>
    %108 = stablehlo.add %103, %107 : tensor<8192xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<8192xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<8192xui32>
    %113 = stablehlo.or %110, %112 : tensor<8192xui32>
    %114 = stablehlo.xor %108, %113 : tensor<8192xui32>
    %115 = stablehlo.add %108, %114 : tensor<8192xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<8192xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<8192xui32>
    %120 = stablehlo.or %117, %119 : tensor<8192xui32>
    %121 = stablehlo.xor %115, %120 : tensor<8192xui32>
    %122 = stablehlo.add %115, %121 : tensor<8192xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<8192xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<8192xui32>
    %127 = stablehlo.or %124, %126 : tensor<8192xui32>
    %128 = stablehlo.xor %122, %127 : tensor<8192xui32>
    %129 = stablehlo.add %122, %128 : tensor<8192xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<8192xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<8192xui32>
    %134 = stablehlo.or %131, %133 : tensor<8192xui32>
    %135 = stablehlo.xor %129, %134 : tensor<8192xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %137 = stablehlo.add %129, %136 : tensor<8192xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %139 = stablehlo.add %135, %138 : tensor<8192xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %141 = stablehlo.add %139, %140 : tensor<8192xui32>
    %142 = stablehlo.add %137, %141 : tensor<8192xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<8192xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<8192xui32>
    %147 = stablehlo.or %144, %146 : tensor<8192xui32>
    %148 = stablehlo.xor %142, %147 : tensor<8192xui32>
    %149 = stablehlo.add %142, %148 : tensor<8192xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<8192xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<8192xui32>
    %154 = stablehlo.or %151, %153 : tensor<8192xui32>
    %155 = stablehlo.xor %149, %154 : tensor<8192xui32>
    %156 = stablehlo.add %149, %155 : tensor<8192xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<8192xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<8192xui32>
    %161 = stablehlo.or %158, %160 : tensor<8192xui32>
    %162 = stablehlo.xor %156, %161 : tensor<8192xui32>
    %163 = stablehlo.add %156, %162 : tensor<8192xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<8192xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<8192xui32>
    %168 = stablehlo.or %165, %167 : tensor<8192xui32>
    %169 = stablehlo.xor %163, %168 : tensor<8192xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %171 = stablehlo.add %163, %170 : tensor<8192xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %173 = stablehlo.add %169, %172 : tensor<8192xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<8192xui32>
    %175 = stablehlo.add %173, %174 : tensor<8192xui32>
    return %171, %175 : tensor<8192xui32>, tensor<8192xui32>
  }
  func.func private @_normal_5(%arg0: tensor<2xui32>) -> tensor<32x512x16xf32> {
    %0 = call @_normal_real_6(%arg0) : (tensor<2xui32>) -> tensor<32x512x16xf32>
    return %0 : tensor<32x512x16xf32>
  }
  func.func private @_normal_real_6(%arg0: tensor<2xui32>) -> tensor<32x512x16xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x512x16xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x512x16xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x512x16xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x512x16xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x512x16xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x512x16xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x512x16xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x512x16xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x512x16xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x512x16xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x512x16xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x512x16xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x512x16xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x512x16xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x512x16xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x512x16xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x512x16xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x512x16xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x512x16xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x512x16xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x512x16xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x512x16xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x512x16xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_7(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x512x16xf32>
    %1 = stablehlo.negate %0 : tensor<32x512x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x512x16xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x512x16xf32>
    %4 = stablehlo.negate %3 : tensor<32x512x16xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x512x16xf32>, tensor<32x512x16xf32>) -> tensor<32x512x16xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x512x16xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x512x16xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x512x16xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x512x16xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x512x16xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x512x16xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x512x16xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x512x16xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x512x16xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x512x16xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x512x16xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x512x16xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x512x16xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x512x16xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x512x16xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x512x16xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x512x16xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x512x16xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x512x16xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x512x16xf32>
    %36 = stablehlo.abs %0 : tensor<32x512x16xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x512x16xf32>, tensor<32x512x16xf32>) -> tensor<32x512x16xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x512x16xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x512x16xi1>, tensor<32x512x16xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x512x16xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x512x16xf32>
    return %41 : tensor<32x512x16xf32>
  }
  func.func private @_uniform_7(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x512x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<262144xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:131072] : (tensor<262144xui32>) -> tensor<131072xui32>
    %8 = stablehlo.slice %2 [131072:262144] : (tensor<262144xui32>) -> tensor<131072xui32>
    %9:2 = call @threefry2x32_8(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<131072xui32>, tensor<131072xui32>) -> (tensor<131072xui32>, tensor<131072xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<131072xui32>, tensor<131072xui32>) -> tensor<262144xui32>
    %11 = stablehlo.reshape %10 : (tensor<262144xui32>) -> tensor<32x512x16xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x512x16xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x512x16xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x512x16xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x512x16xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x512x16xui32>) -> tensor<32x512x16xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x512x16xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x512x16xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x512x16xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x512x16xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x512x16xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x512x16xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x512x16xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x512x16xf32>
    return %25 : tensor<32x512x16xf32>
  }
  func.func private @threefry2x32_8(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<131072xui32>, %arg3: tensor<131072xui32>) -> (tensor<131072xui32>, tensor<131072xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<131072xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<131072xui32>
    %6 = stablehlo.add %3, %5 : tensor<131072xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<131072xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<131072xui32>
    %11 = stablehlo.or %8, %10 : tensor<131072xui32>
    %12 = stablehlo.xor %6, %11 : tensor<131072xui32>
    %13 = stablehlo.add %6, %12 : tensor<131072xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<131072xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<131072xui32>
    %18 = stablehlo.or %15, %17 : tensor<131072xui32>
    %19 = stablehlo.xor %13, %18 : tensor<131072xui32>
    %20 = stablehlo.add %13, %19 : tensor<131072xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<131072xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<131072xui32>
    %25 = stablehlo.or %22, %24 : tensor<131072xui32>
    %26 = stablehlo.xor %20, %25 : tensor<131072xui32>
    %27 = stablehlo.add %20, %26 : tensor<131072xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<131072xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<131072xui32>
    %32 = stablehlo.or %29, %31 : tensor<131072xui32>
    %33 = stablehlo.xor %27, %32 : tensor<131072xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %35 = stablehlo.add %27, %34 : tensor<131072xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %37 = stablehlo.add %33, %36 : tensor<131072xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %39 = stablehlo.add %37, %38 : tensor<131072xui32>
    %40 = stablehlo.add %35, %39 : tensor<131072xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<131072xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<131072xui32>
    %45 = stablehlo.or %42, %44 : tensor<131072xui32>
    %46 = stablehlo.xor %40, %45 : tensor<131072xui32>
    %47 = stablehlo.add %40, %46 : tensor<131072xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<131072xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<131072xui32>
    %52 = stablehlo.or %49, %51 : tensor<131072xui32>
    %53 = stablehlo.xor %47, %52 : tensor<131072xui32>
    %54 = stablehlo.add %47, %53 : tensor<131072xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<131072xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<131072xui32>
    %59 = stablehlo.or %56, %58 : tensor<131072xui32>
    %60 = stablehlo.xor %54, %59 : tensor<131072xui32>
    %61 = stablehlo.add %54, %60 : tensor<131072xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<131072xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<131072xui32>
    %66 = stablehlo.or %63, %65 : tensor<131072xui32>
    %67 = stablehlo.xor %61, %66 : tensor<131072xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %69 = stablehlo.add %61, %68 : tensor<131072xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %71 = stablehlo.add %67, %70 : tensor<131072xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %73 = stablehlo.add %71, %72 : tensor<131072xui32>
    %74 = stablehlo.add %69, %73 : tensor<131072xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<131072xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<131072xui32>
    %79 = stablehlo.or %76, %78 : tensor<131072xui32>
    %80 = stablehlo.xor %74, %79 : tensor<131072xui32>
    %81 = stablehlo.add %74, %80 : tensor<131072xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<131072xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<131072xui32>
    %86 = stablehlo.or %83, %85 : tensor<131072xui32>
    %87 = stablehlo.xor %81, %86 : tensor<131072xui32>
    %88 = stablehlo.add %81, %87 : tensor<131072xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<131072xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<131072xui32>
    %93 = stablehlo.or %90, %92 : tensor<131072xui32>
    %94 = stablehlo.xor %88, %93 : tensor<131072xui32>
    %95 = stablehlo.add %88, %94 : tensor<131072xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<131072xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<131072xui32>
    %100 = stablehlo.or %97, %99 : tensor<131072xui32>
    %101 = stablehlo.xor %95, %100 : tensor<131072xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %103 = stablehlo.add %95, %102 : tensor<131072xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %105 = stablehlo.add %101, %104 : tensor<131072xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %107 = stablehlo.add %105, %106 : tensor<131072xui32>
    %108 = stablehlo.add %103, %107 : tensor<131072xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<131072xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<131072xui32>
    %113 = stablehlo.or %110, %112 : tensor<131072xui32>
    %114 = stablehlo.xor %108, %113 : tensor<131072xui32>
    %115 = stablehlo.add %108, %114 : tensor<131072xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<131072xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<131072xui32>
    %120 = stablehlo.or %117, %119 : tensor<131072xui32>
    %121 = stablehlo.xor %115, %120 : tensor<131072xui32>
    %122 = stablehlo.add %115, %121 : tensor<131072xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<131072xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<131072xui32>
    %127 = stablehlo.or %124, %126 : tensor<131072xui32>
    %128 = stablehlo.xor %122, %127 : tensor<131072xui32>
    %129 = stablehlo.add %122, %128 : tensor<131072xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<131072xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<131072xui32>
    %134 = stablehlo.or %131, %133 : tensor<131072xui32>
    %135 = stablehlo.xor %129, %134 : tensor<131072xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %137 = stablehlo.add %129, %136 : tensor<131072xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %139 = stablehlo.add %135, %138 : tensor<131072xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %141 = stablehlo.add %139, %140 : tensor<131072xui32>
    %142 = stablehlo.add %137, %141 : tensor<131072xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<131072xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<131072xui32>
    %147 = stablehlo.or %144, %146 : tensor<131072xui32>
    %148 = stablehlo.xor %142, %147 : tensor<131072xui32>
    %149 = stablehlo.add %142, %148 : tensor<131072xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<131072xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<131072xui32>
    %154 = stablehlo.or %151, %153 : tensor<131072xui32>
    %155 = stablehlo.xor %149, %154 : tensor<131072xui32>
    %156 = stablehlo.add %149, %155 : tensor<131072xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<131072xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<131072xui32>
    %161 = stablehlo.or %158, %160 : tensor<131072xui32>
    %162 = stablehlo.xor %156, %161 : tensor<131072xui32>
    %163 = stablehlo.add %156, %162 : tensor<131072xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<131072xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<131072xui32>
    %168 = stablehlo.or %165, %167 : tensor<131072xui32>
    %169 = stablehlo.xor %163, %168 : tensor<131072xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %171 = stablehlo.add %163, %170 : tensor<131072xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %173 = stablehlo.add %169, %172 : tensor<131072xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<131072xui32>
    %175 = stablehlo.add %173, %174 : tensor<131072xui32>
    return %171, %175 : tensor<131072xui32>, tensor<131072xui32>
  }
  func.func private @_normal_9(%arg0: tensor<2xui32>) -> tensor<32x16xf32> {
    %0 = call @_normal_real_10(%arg0) : (tensor<2xui32>) -> tensor<32x16xf32>
    return %0 : tensor<32x16xf32>
  }
  func.func private @_normal_real_10(%arg0: tensor<2xui32>) -> tensor<32x16xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x16xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x16xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x16xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x16xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x16xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x16xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x16xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x16xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x16xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x16xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x16xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x16xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x16xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x16xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x16xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x16xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x16xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x16xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x16xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x16xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x16xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x16xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x16xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_11(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x16xf32>
    %1 = stablehlo.negate %0 : tensor<32x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x16xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x16xf32>
    %4 = stablehlo.negate %3 : tensor<32x16xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x16xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x16xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x16xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x16xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x16xi1>, tensor<32x16xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x16xi1>, tensor<32x16xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x16xi1>, tensor<32x16xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x16xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x16xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x16xi1>, tensor<32x16xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x16xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x16xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x16xi1>, tensor<32x16xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x16xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x16xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x16xi1>, tensor<32x16xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x16xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x16xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x16xi1>, tensor<32x16xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x16xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x16xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x16xi1>, tensor<32x16xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x16xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x16xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x16xi1>, tensor<32x16xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x16xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x16xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x16xi1>, tensor<32x16xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x16xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x16xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x16xf32>
    %36 = stablehlo.abs %0 : tensor<32x16xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x16xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x16xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x16xi1>, tensor<32x16xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x16xf32>
    return %41 : tensor<32x16xf32>
  }
  func.func private @_uniform_11(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<512xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:256] : (tensor<512xui32>) -> tensor<256xui32>
    %8 = stablehlo.slice %2 [256:512] : (tensor<512xui32>) -> tensor<256xui32>
    %9:2 = call @threefry2x32_12(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<256xui32>, tensor<256xui32>) -> (tensor<256xui32>, tensor<256xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<256xui32>, tensor<256xui32>) -> tensor<512xui32>
    %11 = stablehlo.reshape %10 : (tensor<512xui32>) -> tensor<32x16xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x16xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x16xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x16xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x16xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x16xui32>) -> tensor<32x16xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x16xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x16xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x16xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x16xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x16xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x16xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x16xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x16xf32>
    return %25 : tensor<32x16xf32>
  }
  func.func private @threefry2x32_12(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<256xui32>, %arg3: tensor<256xui32>) -> (tensor<256xui32>, tensor<256xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<256xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<256xui32>
    %6 = stablehlo.add %3, %5 : tensor<256xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<256xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<256xui32>
    %11 = stablehlo.or %8, %10 : tensor<256xui32>
    %12 = stablehlo.xor %6, %11 : tensor<256xui32>
    %13 = stablehlo.add %6, %12 : tensor<256xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<256xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<256xui32>
    %18 = stablehlo.or %15, %17 : tensor<256xui32>
    %19 = stablehlo.xor %13, %18 : tensor<256xui32>
    %20 = stablehlo.add %13, %19 : tensor<256xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<256xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<256xui32>
    %25 = stablehlo.or %22, %24 : tensor<256xui32>
    %26 = stablehlo.xor %20, %25 : tensor<256xui32>
    %27 = stablehlo.add %20, %26 : tensor<256xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<256xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<256xui32>
    %32 = stablehlo.or %29, %31 : tensor<256xui32>
    %33 = stablehlo.xor %27, %32 : tensor<256xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %35 = stablehlo.add %27, %34 : tensor<256xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %37 = stablehlo.add %33, %36 : tensor<256xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %39 = stablehlo.add %37, %38 : tensor<256xui32>
    %40 = stablehlo.add %35, %39 : tensor<256xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<256xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<256xui32>
    %45 = stablehlo.or %42, %44 : tensor<256xui32>
    %46 = stablehlo.xor %40, %45 : tensor<256xui32>
    %47 = stablehlo.add %40, %46 : tensor<256xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<256xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<256xui32>
    %52 = stablehlo.or %49, %51 : tensor<256xui32>
    %53 = stablehlo.xor %47, %52 : tensor<256xui32>
    %54 = stablehlo.add %47, %53 : tensor<256xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<256xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<256xui32>
    %59 = stablehlo.or %56, %58 : tensor<256xui32>
    %60 = stablehlo.xor %54, %59 : tensor<256xui32>
    %61 = stablehlo.add %54, %60 : tensor<256xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<256xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<256xui32>
    %66 = stablehlo.or %63, %65 : tensor<256xui32>
    %67 = stablehlo.xor %61, %66 : tensor<256xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %69 = stablehlo.add %61, %68 : tensor<256xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %71 = stablehlo.add %67, %70 : tensor<256xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %73 = stablehlo.add %71, %72 : tensor<256xui32>
    %74 = stablehlo.add %69, %73 : tensor<256xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<256xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<256xui32>
    %79 = stablehlo.or %76, %78 : tensor<256xui32>
    %80 = stablehlo.xor %74, %79 : tensor<256xui32>
    %81 = stablehlo.add %74, %80 : tensor<256xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<256xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<256xui32>
    %86 = stablehlo.or %83, %85 : tensor<256xui32>
    %87 = stablehlo.xor %81, %86 : tensor<256xui32>
    %88 = stablehlo.add %81, %87 : tensor<256xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<256xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<256xui32>
    %93 = stablehlo.or %90, %92 : tensor<256xui32>
    %94 = stablehlo.xor %88, %93 : tensor<256xui32>
    %95 = stablehlo.add %88, %94 : tensor<256xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<256xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<256xui32>
    %100 = stablehlo.or %97, %99 : tensor<256xui32>
    %101 = stablehlo.xor %95, %100 : tensor<256xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %103 = stablehlo.add %95, %102 : tensor<256xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %105 = stablehlo.add %101, %104 : tensor<256xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %107 = stablehlo.add %105, %106 : tensor<256xui32>
    %108 = stablehlo.add %103, %107 : tensor<256xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<256xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<256xui32>
    %113 = stablehlo.or %110, %112 : tensor<256xui32>
    %114 = stablehlo.xor %108, %113 : tensor<256xui32>
    %115 = stablehlo.add %108, %114 : tensor<256xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<256xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<256xui32>
    %120 = stablehlo.or %117, %119 : tensor<256xui32>
    %121 = stablehlo.xor %115, %120 : tensor<256xui32>
    %122 = stablehlo.add %115, %121 : tensor<256xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<256xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<256xui32>
    %127 = stablehlo.or %124, %126 : tensor<256xui32>
    %128 = stablehlo.xor %122, %127 : tensor<256xui32>
    %129 = stablehlo.add %122, %128 : tensor<256xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<256xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<256xui32>
    %134 = stablehlo.or %131, %133 : tensor<256xui32>
    %135 = stablehlo.xor %129, %134 : tensor<256xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %137 = stablehlo.add %129, %136 : tensor<256xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %139 = stablehlo.add %135, %138 : tensor<256xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %141 = stablehlo.add %139, %140 : tensor<256xui32>
    %142 = stablehlo.add %137, %141 : tensor<256xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<256xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<256xui32>
    %147 = stablehlo.or %144, %146 : tensor<256xui32>
    %148 = stablehlo.xor %142, %147 : tensor<256xui32>
    %149 = stablehlo.add %142, %148 : tensor<256xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<256xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<256xui32>
    %154 = stablehlo.or %151, %153 : tensor<256xui32>
    %155 = stablehlo.xor %149, %154 : tensor<256xui32>
    %156 = stablehlo.add %149, %155 : tensor<256xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<256xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<256xui32>
    %161 = stablehlo.or %158, %160 : tensor<256xui32>
    %162 = stablehlo.xor %156, %161 : tensor<256xui32>
    %163 = stablehlo.add %156, %162 : tensor<256xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<256xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<256xui32>
    %168 = stablehlo.or %165, %167 : tensor<256xui32>
    %169 = stablehlo.xor %163, %168 : tensor<256xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %171 = stablehlo.add %163, %170 : tensor<256xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %173 = stablehlo.add %169, %172 : tensor<256xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<256xui32>
    %175 = stablehlo.add %173, %174 : tensor<256xui32>
    return %171, %175 : tensor<256xui32>, tensor<256xui32>
  }
  func.func private @_normal_13(%arg0: tensor<2xui32>) -> tensor<32x1x16xf32> {
    %0 = call @_normal_real_14(%arg0) : (tensor<2xui32>) -> tensor<32x1x16xf32>
    return %0 : tensor<32x1x16xf32>
  }
  func.func private @_normal_real_14(%arg0: tensor<2xui32>) -> tensor<32x1x16xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x1x16xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x1x16xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x1x16xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x1x16xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x1x16xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x1x16xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x1x16xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x1x16xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x1x16xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x1x16xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x1x16xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x1x16xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x1x16xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x1x16xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x1x16xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x1x16xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x1x16xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x1x16xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x1x16xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x1x16xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x1x16xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x1x16xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x1x16xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_15(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x1x16xf32>
    %1 = stablehlo.negate %0 : tensor<32x1x16xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x1x16xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x1x16xf32>
    %4 = stablehlo.negate %3 : tensor<32x1x16xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x1x16xf32>, tensor<32x1x16xf32>) -> tensor<32x1x16xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x1x16xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x1x16xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x1x16xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x1x16xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x1x16xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x1x16xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x1x16xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x1x16xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x1x16xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x1x16xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x1x16xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x1x16xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x1x16xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x1x16xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x1x16xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x1x16xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x1x16xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x1x16xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x1x16xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x1x16xf32>
    %36 = stablehlo.abs %0 : tensor<32x1x16xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x1x16xf32>, tensor<32x1x16xf32>) -> tensor<32x1x16xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x1x16xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x1x16xi1>, tensor<32x1x16xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x1x16xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x1x16xf32>
    return %41 : tensor<32x1x16xf32>
  }
  func.func private @_uniform_15(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x1x16xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<512xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:256] : (tensor<512xui32>) -> tensor<256xui32>
    %8 = stablehlo.slice %2 [256:512] : (tensor<512xui32>) -> tensor<256xui32>
    %9:2 = call @threefry2x32_12(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<256xui32>, tensor<256xui32>) -> (tensor<256xui32>, tensor<256xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<256xui32>, tensor<256xui32>) -> tensor<512xui32>
    %11 = stablehlo.reshape %10 : (tensor<512xui32>) -> tensor<32x1x16xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x1x16xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x1x16xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x1x16xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x1x16xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x1x16xui32>) -> tensor<32x1x16xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x1x16xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x1x16xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x1x16xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x1x16xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x1x16xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x1x16xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x1x16xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x1x16xf32>
    return %25 : tensor<32x1x16xf32>
  }
  func.func private @_normal_16(%arg0: tensor<2xui32>) -> tensor<32x16x512xf32> {
    %0 = call @_normal_real_17(%arg0) : (tensor<2xui32>) -> tensor<32x16x512xf32>
    return %0 : tensor<32x16x512xf32>
  }
  func.func private @_normal_real_17(%arg0: tensor<2xui32>) -> tensor<32x16x512xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x16x512xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x16x512xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x16x512xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x16x512xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x16x512xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x16x512xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x16x512xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x16x512xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x16x512xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x16x512xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x16x512xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x16x512xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x16x512xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x16x512xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x16x512xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x16x512xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x16x512xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x16x512xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x16x512xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x16x512xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x16x512xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x16x512xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x16x512xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_18(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x16x512xf32>
    %1 = stablehlo.negate %0 : tensor<32x16x512xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x16x512xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x16x512xf32>
    %4 = stablehlo.negate %3 : tensor<32x16x512xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x16x512xf32>, tensor<32x16x512xf32>) -> tensor<32x16x512xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x16x512xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x16x512xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x16x512xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x16x512xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x16x512xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x16x512xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x16x512xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x16x512xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x16x512xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x16x512xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x16x512xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x16x512xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x16x512xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x16x512xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x16x512xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x16x512xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x16x512xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x16x512xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x16x512xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x16x512xf32>
    %36 = stablehlo.abs %0 : tensor<32x16x512xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x16x512xf32>, tensor<32x16x512xf32>) -> tensor<32x16x512xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x16x512xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x16x512xi1>, tensor<32x16x512xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x16x512xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x16x512xf32>
    return %41 : tensor<32x16x512xf32>
  }
  func.func private @_uniform_18(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x16x512xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<262144xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:131072] : (tensor<262144xui32>) -> tensor<131072xui32>
    %8 = stablehlo.slice %2 [131072:262144] : (tensor<262144xui32>) -> tensor<131072xui32>
    %9:2 = call @threefry2x32_8(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<131072xui32>, tensor<131072xui32>) -> (tensor<131072xui32>, tensor<131072xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<131072xui32>, tensor<131072xui32>) -> tensor<262144xui32>
    %11 = stablehlo.reshape %10 : (tensor<262144xui32>) -> tensor<32x16x512xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x16x512xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x16x512xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x16x512xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x16x512xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x16x512xui32>) -> tensor<32x16x512xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x16x512xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x16x512xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x16x512xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x16x512xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x16x512xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x16x512xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x16x512xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x16x512xf32>
    return %25 : tensor<32x16x512xf32>
  }
  func.func private @_normal_19(%arg0: tensor<2xui32>) -> tensor<32x512x784xf32> {
    %0 = call @_normal_real_20(%arg0) : (tensor<2xui32>) -> tensor<32x512x784xf32>
    return %0 : tensor<32x512x784xf32>
  }
  func.func private @_normal_real_20(%arg0: tensor<2xui32>) -> tensor<32x512x784xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x512x784xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x512x784xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x512x784xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x512x784xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x512x784xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x512x784xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x512x784xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x512x784xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x512x784xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x512x784xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x512x784xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x512x784xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x512x784xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x512x784xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x512x784xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x512x784xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x512x784xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x512x784xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x512x784xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x512x784xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x512x784xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x512x784xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x512x784xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_21(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x512x784xf32>
    %1 = stablehlo.negate %0 : tensor<32x512x784xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x512x784xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x512x784xf32>
    %4 = stablehlo.negate %3 : tensor<32x512x784xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x512x784xf32>, tensor<32x512x784xf32>) -> tensor<32x512x784xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x512x784xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x512x784xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x512x784xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x512x784xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x512x784xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x512x784xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x512x784xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x512x784xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x512x784xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x512x784xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x512x784xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x512x784xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x512x784xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x512x784xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x512x784xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x512x784xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x512x784xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x512x784xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x512x784xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x512x784xf32>
    %36 = stablehlo.abs %0 : tensor<32x512x784xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x512x784xf32>, tensor<32x512x784xf32>) -> tensor<32x512x784xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x512x784xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x512x784xi1>, tensor<32x512x784xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x512x784xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x512x784xf32>
    return %41 : tensor<32x512x784xf32>
  }
  func.func private @_uniform_21(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x512x784xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<12845056xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:6422528] : (tensor<12845056xui32>) -> tensor<6422528xui32>
    %8 = stablehlo.slice %2 [6422528:12845056] : (tensor<12845056xui32>) -> tensor<6422528xui32>
    %9:2 = call @threefry2x32_0(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<6422528xui32>, tensor<6422528xui32>) -> (tensor<6422528xui32>, tensor<6422528xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<6422528xui32>, tensor<6422528xui32>) -> tensor<12845056xui32>
    %11 = stablehlo.reshape %10 : (tensor<12845056xui32>) -> tensor<32x512x784xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x512x784xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x512x784xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x512x784xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x512x784xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x512x784xui32>) -> tensor<32x512x784xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x512x784xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x512x784xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x512x784xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x512x784xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x512x784xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x512x784xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<32x512x784xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x512x784xf32>
    return %25 : tensor<32x512x784xf32>
  }
  func.func private @_normal_22(%arg0: tensor<2xui32>) -> tensor<32x784xf32> {
    %0 = call @_normal_real_23(%arg0) : (tensor<2xui32>) -> tensor<32x784xf32>
    return %0 : tensor<32x784xf32>
  }
  func.func private @_normal_real_23(%arg0: tensor<2xui32>) -> tensor<32x784xf32> {
    %cst = stablehlo.constant dense<1.41421354> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0x7F800000> : tensor<32x784xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<32x784xf32>
    %cst_2 = stablehlo.constant dense<2.83297682> : tensor<32x784xf32>
    %cst_3 = stablehlo.constant dense<1.50140941> : tensor<32x784xf32>
    %cst_4 = stablehlo.constant dense<1.00167406> : tensor<32x784xf32>
    %cst_5 = stablehlo.constant dense<0.246640727> : tensor<32x784xf32>
    %cst_6 = stablehlo.constant dense<0.00943887047> : tensor<32x784xf32>
    %cst_7 = stablehlo.constant dense<-0.00417768164> : tensor<32x784xf32>
    %cst_8 = stablehlo.constant dense<-0.0076224613> : tensor<32x784xf32>
    %cst_9 = stablehlo.constant dense<-0.00125372503> : tensor<32x784xf32>
    %cst_10 = stablehlo.constant dense<0.00573950773> : tensor<32x784xf32>
    %cst_11 = stablehlo.constant dense<2.1858087E-4> : tensor<32x784xf32>
    %cst_12 = stablehlo.constant dense<-0.00367342844> : tensor<32x784xf32>
    %cst_13 = stablehlo.constant dense<-4.39150654E-6> : tensor<32x784xf32>
    %cst_14 = stablehlo.constant dense<0.00134934322> : tensor<32x784xf32>
    %cst_15 = stablehlo.constant dense<-3.5233877E-6> : tensor<32x784xf32>
    %cst_16 = stablehlo.constant dense<1.00950558E-4> : tensor<32x784xf32>
    %cst_17 = stablehlo.constant dense<3.43273939E-7> : tensor<32x784xf32>
    %cst_18 = stablehlo.constant dense<-2.00214257E-4> : tensor<32x784xf32>
    %cst_19 = stablehlo.constant dense<2.81022636E-8> : tensor<32x784xf32>
    %cst_20 = stablehlo.constant dense<3.000000e+00> : tensor<32x784xf32>
    %cst_21 = stablehlo.constant dense<2.500000e+00> : tensor<32x784xf32>
    %cst_22 = stablehlo.constant dense<5.000000e+00> : tensor<32x784xf32>
    %cst_23 = stablehlo.constant dense<-0.99999994> : tensor<f32>
    %cst_24 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = call @_uniform_24(%arg0, %cst_23, %cst_24) : (tensor<2xui32>, tensor<f32>, tensor<f32>) -> tensor<32x784xf32>
    %1 = stablehlo.negate %0 : tensor<32x784xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<32x784xf32>
    %3 = stablehlo.log_plus_one %2 : tensor<32x784xf32>
    %4 = stablehlo.negate %3 : tensor<32x784xf32>
    %5 = stablehlo.compare  LT, %4, %cst_22 : (tensor<32x784xf32>, tensor<32x784xf32>) -> tensor<32x784xi1>
    %6 = stablehlo.subtract %4, %cst_21 : tensor<32x784xf32>
    %7 = stablehlo.sqrt %4 : tensor<32x784xf32>
    %8 = stablehlo.subtract %7, %cst_20 : tensor<32x784xf32>
    %9 = stablehlo.select %5, %6, %8 : tensor<32x784xi1>, tensor<32x784xf32>
    %10 = stablehlo.select %5, %cst_19, %cst_18 : tensor<32x784xi1>, tensor<32x784xf32>
    %11 = stablehlo.select %5, %cst_17, %cst_16 : tensor<32x784xi1>, tensor<32x784xf32>
    %12 = stablehlo.multiply %10, %9 : tensor<32x784xf32>
    %13 = stablehlo.add %11, %12 : tensor<32x784xf32>
    %14 = stablehlo.select %5, %cst_15, %cst_14 : tensor<32x784xi1>, tensor<32x784xf32>
    %15 = stablehlo.multiply %13, %9 : tensor<32x784xf32>
    %16 = stablehlo.add %14, %15 : tensor<32x784xf32>
    %17 = stablehlo.select %5, %cst_13, %cst_12 : tensor<32x784xi1>, tensor<32x784xf32>
    %18 = stablehlo.multiply %16, %9 : tensor<32x784xf32>
    %19 = stablehlo.add %17, %18 : tensor<32x784xf32>
    %20 = stablehlo.select %5, %cst_11, %cst_10 : tensor<32x784xi1>, tensor<32x784xf32>
    %21 = stablehlo.multiply %19, %9 : tensor<32x784xf32>
    %22 = stablehlo.add %20, %21 : tensor<32x784xf32>
    %23 = stablehlo.select %5, %cst_9, %cst_8 : tensor<32x784xi1>, tensor<32x784xf32>
    %24 = stablehlo.multiply %22, %9 : tensor<32x784xf32>
    %25 = stablehlo.add %23, %24 : tensor<32x784xf32>
    %26 = stablehlo.select %5, %cst_7, %cst_6 : tensor<32x784xi1>, tensor<32x784xf32>
    %27 = stablehlo.multiply %25, %9 : tensor<32x784xf32>
    %28 = stablehlo.add %26, %27 : tensor<32x784xf32>
    %29 = stablehlo.select %5, %cst_5, %cst_4 : tensor<32x784xi1>, tensor<32x784xf32>
    %30 = stablehlo.multiply %28, %9 : tensor<32x784xf32>
    %31 = stablehlo.add %29, %30 : tensor<32x784xf32>
    %32 = stablehlo.select %5, %cst_3, %cst_2 : tensor<32x784xi1>, tensor<32x784xf32>
    %33 = stablehlo.multiply %31, %9 : tensor<32x784xf32>
    %34 = stablehlo.add %32, %33 : tensor<32x784xf32>
    %35 = stablehlo.multiply %34, %0 : tensor<32x784xf32>
    %36 = stablehlo.abs %0 : tensor<32x784xf32>
    %37 = stablehlo.compare  EQ, %36, %cst_1 : (tensor<32x784xf32>, tensor<32x784xf32>) -> tensor<32x784xi1>
    %38 = stablehlo.multiply %0, %cst_0 : tensor<32x784xf32>
    %39 = stablehlo.select %37, %38, %35 : tensor<32x784xi1>, tensor<32x784xf32>
    %40 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x784xf32>
    %41 = stablehlo.multiply %40, %39 : tensor<32x784xf32>
    return %41 : tensor<32x784xf32>
  }
  func.func private @_uniform_24(%arg0: tensor<2xui32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<32x784xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<1065353216> : tensor<ui32>
    %c_0 = stablehlo.constant dense<9> : tensor<ui32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %2 = stablehlo.iota dim = 0 : tensor<25088xui32>
    %3 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %4 = stablehlo.reshape %3 : (tensor<1xui32>) -> tensor<ui32>
    %5 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %2 [0:12544] : (tensor<25088xui32>) -> tensor<12544xui32>
    %8 = stablehlo.slice %2 [12544:25088] : (tensor<25088xui32>) -> tensor<12544xui32>
    %9:2 = call @threefry2x32_25(%4, %6, %7, %8) : (tensor<ui32>, tensor<ui32>, tensor<12544xui32>, tensor<12544xui32>) -> (tensor<12544xui32>, tensor<12544xui32>)
    %10 = stablehlo.concatenate %9#0, %9#1, dim = 0 : (tensor<12544xui32>, tensor<12544xui32>) -> tensor<25088xui32>
    %11 = stablehlo.reshape %10 : (tensor<25088xui32>) -> tensor<32x784xui32>
    %12 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<32x784xui32>
    %13 = stablehlo.shift_right_logical %11, %12 : tensor<32x784xui32>
    %14 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<32x784xui32>
    %15 = stablehlo.or %13, %14 : tensor<32x784xui32>
    %16 = stablehlo.bitcast_convert %15 : (tensor<32x784xui32>) -> tensor<32x784xf32>
    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x784xf32>
    %18 = stablehlo.subtract %16, %17 : tensor<32x784xf32>
    %19 = stablehlo.subtract %1, %0 : tensor<1x1xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x784xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<32x784xf32>
    %22 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x784xf32>
    %23 = stablehlo.add %21, %22 : tensor<32x784xf32>
    %24 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x784xf32>
    %25 = stablehlo.maximum %24, %23 : tensor<32x784xf32>
    return %25 : tensor<32x784xf32>
  }
  func.func private @threefry2x32_25(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<12544xui32>, %arg3: tensor<12544xui32>) -> (tensor<12544xui32>, tensor<12544xui32>) {
    %c = stablehlo.constant dense<5> : tensor<ui32>
    %c_0 = stablehlo.constant dense<4> : tensor<ui32>
    %c_1 = stablehlo.constant dense<2> : tensor<ui32>
    %c_2 = stablehlo.constant dense<8> : tensor<ui32>
    %c_3 = stablehlo.constant dense<24> : tensor<ui32>
    %c_4 = stablehlo.constant dense<16> : tensor<ui32>
    %c_5 = stablehlo.constant dense<3> : tensor<ui32>
    %c_6 = stablehlo.constant dense<29> : tensor<ui32>
    %c_7 = stablehlo.constant dense<1> : tensor<ui32>
    %c_8 = stablehlo.constant dense<6> : tensor<ui32>
    %c_9 = stablehlo.constant dense<26> : tensor<ui32>
    %c_10 = stablehlo.constant dense<17> : tensor<ui32>
    %c_11 = stablehlo.constant dense<15> : tensor<ui32>
    %c_12 = stablehlo.constant dense<19> : tensor<ui32>
    %c_13 = stablehlo.constant dense<13> : tensor<ui32>
    %c_14 = stablehlo.constant dense<466688986> : tensor<ui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %1 = stablehlo.xor %0, %c_14 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<12544xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<12544xui32>
    %6 = stablehlo.add %3, %5 : tensor<12544xui32>
    %7 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %8 = stablehlo.shift_left %5, %7 : tensor<12544xui32>
    %9 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %10 = stablehlo.shift_right_logical %5, %9 : tensor<12544xui32>
    %11 = stablehlo.or %8, %10 : tensor<12544xui32>
    %12 = stablehlo.xor %6, %11 : tensor<12544xui32>
    %13 = stablehlo.add %6, %12 : tensor<12544xui32>
    %14 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %15 = stablehlo.shift_left %12, %14 : tensor<12544xui32>
    %16 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %17 = stablehlo.shift_right_logical %12, %16 : tensor<12544xui32>
    %18 = stablehlo.or %15, %17 : tensor<12544xui32>
    %19 = stablehlo.xor %13, %18 : tensor<12544xui32>
    %20 = stablehlo.add %13, %19 : tensor<12544xui32>
    %21 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %22 = stablehlo.shift_left %19, %21 : tensor<12544xui32>
    %23 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %24 = stablehlo.shift_right_logical %19, %23 : tensor<12544xui32>
    %25 = stablehlo.or %22, %24 : tensor<12544xui32>
    %26 = stablehlo.xor %20, %25 : tensor<12544xui32>
    %27 = stablehlo.add %20, %26 : tensor<12544xui32>
    %28 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %29 = stablehlo.shift_left %26, %28 : tensor<12544xui32>
    %30 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %31 = stablehlo.shift_right_logical %26, %30 : tensor<12544xui32>
    %32 = stablehlo.or %29, %31 : tensor<12544xui32>
    %33 = stablehlo.xor %27, %32 : tensor<12544xui32>
    %34 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %35 = stablehlo.add %27, %34 : tensor<12544xui32>
    %36 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %37 = stablehlo.add %33, %36 : tensor<12544xui32>
    %38 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %39 = stablehlo.add %37, %38 : tensor<12544xui32>
    %40 = stablehlo.add %35, %39 : tensor<12544xui32>
    %41 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %42 = stablehlo.shift_left %39, %41 : tensor<12544xui32>
    %43 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %44 = stablehlo.shift_right_logical %39, %43 : tensor<12544xui32>
    %45 = stablehlo.or %42, %44 : tensor<12544xui32>
    %46 = stablehlo.xor %40, %45 : tensor<12544xui32>
    %47 = stablehlo.add %40, %46 : tensor<12544xui32>
    %48 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %49 = stablehlo.shift_left %46, %48 : tensor<12544xui32>
    %50 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %51 = stablehlo.shift_right_logical %46, %50 : tensor<12544xui32>
    %52 = stablehlo.or %49, %51 : tensor<12544xui32>
    %53 = stablehlo.xor %47, %52 : tensor<12544xui32>
    %54 = stablehlo.add %47, %53 : tensor<12544xui32>
    %55 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %56 = stablehlo.shift_left %53, %55 : tensor<12544xui32>
    %57 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %58 = stablehlo.shift_right_logical %53, %57 : tensor<12544xui32>
    %59 = stablehlo.or %56, %58 : tensor<12544xui32>
    %60 = stablehlo.xor %54, %59 : tensor<12544xui32>
    %61 = stablehlo.add %54, %60 : tensor<12544xui32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %63 = stablehlo.shift_left %60, %62 : tensor<12544xui32>
    %64 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %65 = stablehlo.shift_right_logical %60, %64 : tensor<12544xui32>
    %66 = stablehlo.or %63, %65 : tensor<12544xui32>
    %67 = stablehlo.xor %61, %66 : tensor<12544xui32>
    %68 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %69 = stablehlo.add %61, %68 : tensor<12544xui32>
    %70 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %71 = stablehlo.add %67, %70 : tensor<12544xui32>
    %72 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %73 = stablehlo.add %71, %72 : tensor<12544xui32>
    %74 = stablehlo.add %69, %73 : tensor<12544xui32>
    %75 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %76 = stablehlo.shift_left %73, %75 : tensor<12544xui32>
    %77 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %78 = stablehlo.shift_right_logical %73, %77 : tensor<12544xui32>
    %79 = stablehlo.or %76, %78 : tensor<12544xui32>
    %80 = stablehlo.xor %74, %79 : tensor<12544xui32>
    %81 = stablehlo.add %74, %80 : tensor<12544xui32>
    %82 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %83 = stablehlo.shift_left %80, %82 : tensor<12544xui32>
    %84 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %85 = stablehlo.shift_right_logical %80, %84 : tensor<12544xui32>
    %86 = stablehlo.or %83, %85 : tensor<12544xui32>
    %87 = stablehlo.xor %81, %86 : tensor<12544xui32>
    %88 = stablehlo.add %81, %87 : tensor<12544xui32>
    %89 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %90 = stablehlo.shift_left %87, %89 : tensor<12544xui32>
    %91 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %92 = stablehlo.shift_right_logical %87, %91 : tensor<12544xui32>
    %93 = stablehlo.or %90, %92 : tensor<12544xui32>
    %94 = stablehlo.xor %88, %93 : tensor<12544xui32>
    %95 = stablehlo.add %88, %94 : tensor<12544xui32>
    %96 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %97 = stablehlo.shift_left %94, %96 : tensor<12544xui32>
    %98 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %99 = stablehlo.shift_right_logical %94, %98 : tensor<12544xui32>
    %100 = stablehlo.or %97, %99 : tensor<12544xui32>
    %101 = stablehlo.xor %95, %100 : tensor<12544xui32>
    %102 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %103 = stablehlo.add %95, %102 : tensor<12544xui32>
    %104 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %105 = stablehlo.add %101, %104 : tensor<12544xui32>
    %106 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %107 = stablehlo.add %105, %106 : tensor<12544xui32>
    %108 = stablehlo.add %103, %107 : tensor<12544xui32>
    %109 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %110 = stablehlo.shift_left %107, %109 : tensor<12544xui32>
    %111 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %112 = stablehlo.shift_right_logical %107, %111 : tensor<12544xui32>
    %113 = stablehlo.or %110, %112 : tensor<12544xui32>
    %114 = stablehlo.xor %108, %113 : tensor<12544xui32>
    %115 = stablehlo.add %108, %114 : tensor<12544xui32>
    %116 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %117 = stablehlo.shift_left %114, %116 : tensor<12544xui32>
    %118 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %119 = stablehlo.shift_right_logical %114, %118 : tensor<12544xui32>
    %120 = stablehlo.or %117, %119 : tensor<12544xui32>
    %121 = stablehlo.xor %115, %120 : tensor<12544xui32>
    %122 = stablehlo.add %115, %121 : tensor<12544xui32>
    %123 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %124 = stablehlo.shift_left %121, %123 : tensor<12544xui32>
    %125 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %126 = stablehlo.shift_right_logical %121, %125 : tensor<12544xui32>
    %127 = stablehlo.or %124, %126 : tensor<12544xui32>
    %128 = stablehlo.xor %122, %127 : tensor<12544xui32>
    %129 = stablehlo.add %122, %128 : tensor<12544xui32>
    %130 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %131 = stablehlo.shift_left %128, %130 : tensor<12544xui32>
    %132 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %133 = stablehlo.shift_right_logical %128, %132 : tensor<12544xui32>
    %134 = stablehlo.or %131, %133 : tensor<12544xui32>
    %135 = stablehlo.xor %129, %134 : tensor<12544xui32>
    %136 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %137 = stablehlo.add %129, %136 : tensor<12544xui32>
    %138 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %139 = stablehlo.add %135, %138 : tensor<12544xui32>
    %140 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %141 = stablehlo.add %139, %140 : tensor<12544xui32>
    %142 = stablehlo.add %137, %141 : tensor<12544xui32>
    %143 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %144 = stablehlo.shift_left %141, %143 : tensor<12544xui32>
    %145 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %146 = stablehlo.shift_right_logical %141, %145 : tensor<12544xui32>
    %147 = stablehlo.or %144, %146 : tensor<12544xui32>
    %148 = stablehlo.xor %142, %147 : tensor<12544xui32>
    %149 = stablehlo.add %142, %148 : tensor<12544xui32>
    %150 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %151 = stablehlo.shift_left %148, %150 : tensor<12544xui32>
    %152 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %153 = stablehlo.shift_right_logical %148, %152 : tensor<12544xui32>
    %154 = stablehlo.or %151, %153 : tensor<12544xui32>
    %155 = stablehlo.xor %149, %154 : tensor<12544xui32>
    %156 = stablehlo.add %149, %155 : tensor<12544xui32>
    %157 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %158 = stablehlo.shift_left %155, %157 : tensor<12544xui32>
    %159 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %160 = stablehlo.shift_right_logical %155, %159 : tensor<12544xui32>
    %161 = stablehlo.or %158, %160 : tensor<12544xui32>
    %162 = stablehlo.xor %156, %161 : tensor<12544xui32>
    %163 = stablehlo.add %156, %162 : tensor<12544xui32>
    %164 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %165 = stablehlo.shift_left %162, %164 : tensor<12544xui32>
    %166 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %167 = stablehlo.shift_right_logical %162, %166 : tensor<12544xui32>
    %168 = stablehlo.or %165, %167 : tensor<12544xui32>
    %169 = stablehlo.xor %163, %168 : tensor<12544xui32>
    %170 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %171 = stablehlo.add %163, %170 : tensor<12544xui32>
    %172 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %173 = stablehlo.add %169, %172 : tensor<12544xui32>
    %174 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<12544xui32>
    %175 = stablehlo.add %173, %174 : tensor<12544xui32>
    return %171, %175 : tensor<12544xui32>, tensor<12544xui32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
}
