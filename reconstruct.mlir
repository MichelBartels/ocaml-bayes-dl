module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<784x512xf32>, %arg1: tensor<784x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512x16xf32>, %arg5: tensor<512x16xf32>, %arg6: tensor<16xf32>, %arg7: tensor<16xf32>, %arg8: tensor<512x16xf32>, %arg9: tensor<512x16xf32>, %arg10: tensor<16xf32>, %arg11: tensor<16xf32>, %arg12: tensor<16x512xf32>, %arg13: tensor<16x512xf32>, %arg14: tensor<512xf32>, %arg15: tensor<512xf32>, %arg16: tensor<512x784xf32>, %arg17: tensor<512x784xf32>, %arg18: tensor<784xf32>, %arg19: tensor<784xf32>, %arg20: tensor<128x784xf32>) -> (tensor<f32> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %cst_1 = stablehlo.constant dense<7.840000e+02> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %cst_3 = stablehlo.constant dense<-5.000000e-01> : tensor<f32>
    %cst_4 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<128x784x512xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<128x512xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<128x512x16xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<128x16xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<128x1x16xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<128x16x512xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<128x512x784xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<128x784xf32>
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x784x512xf32>) -> tensor<128x784x512xf32>
    %2 = stablehlo.multiply %1, %cst_7 : tensor<128x784x512xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<784x512xf32>) -> tensor<1x784x512xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x784x512xf32>) -> tensor<128x784x512xf32>
    %5 = stablehlo.add %4, %2 : tensor<128x784x512xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %8 = stablehlo.multiply %7, %cst_8 : tensor<128x512xf32>
    %9 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %11 = stablehlo.add %10, %8 : tensor<128x512xf32>
    %12 = stablehlo.broadcast_in_dim %arg20, dims = [0, 2] : (tensor<128x784xf32>) -> tensor<128x1x784xf32>
    %13 = stablehlo.dot_general %12, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x784xf32>, tensor<128x784x512xf32>) -> tensor<128x1x512xf32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 2] : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %15 = stablehlo.add %13, %14 : tensor<128x1x512xf32>
    %16 = stablehlo.tanh %15 : tensor<128x1x512xf32>
    %17 = stablehlo.broadcast_in_dim %arg5, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %19 = stablehlo.multiply %18, %cst_9 : tensor<128x512x16xf32>
    %20 = stablehlo.broadcast_in_dim %arg4, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %22 = stablehlo.add %21, %19 : tensor<128x512x16xf32>
    %23 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %25 = stablehlo.multiply %24, %cst_10 : tensor<128x16xf32>
    %26 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %28 = stablehlo.add %27, %25 : tensor<128x16xf32>
    %29 = stablehlo.dot_general %16, %22, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x512xf32>, tensor<128x512x16xf32>) -> tensor<128x1x16xf32>
    %30 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
    %31 = stablehlo.add %29, %30 : tensor<128x1x16xf32>
    %32 = stablehlo.broadcast_in_dim %arg9, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %34 = stablehlo.multiply %33, %cst_9 : tensor<128x512x16xf32>
    %35 = stablehlo.broadcast_in_dim %arg8, dims = [1, 2] : (tensor<512x16xf32>) -> tensor<1x512x16xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2] : (tensor<1x512x16xf32>) -> tensor<128x512x16xf32>
    %37 = stablehlo.add %36, %34 : tensor<128x512x16xf32>
    %38 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %40 = stablehlo.multiply %39, %cst_10 : tensor<128x16xf32>
    %41 = stablehlo.broadcast_in_dim %arg10, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<128x16xf32>
    %43 = stablehlo.add %42, %40 : tensor<128x16xf32>
    %44 = stablehlo.dot_general %16, %37, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x512xf32>, tensor<128x512x16xf32>) -> tensor<128x1x16xf32>
    %45 = stablehlo.broadcast_in_dim %43, dims = [0, 2] : (tensor<128x16xf32>) -> tensor<128x1x16xf32>
    %46 = stablehlo.add %44, %45 : tensor<128x1x16xf32>
    %47 = stablehlo.exponential %46 : tensor<128x1x16xf32>
    %48 = stablehlo.multiply %47, %cst_11 : tensor<128x1x16xf32>
    %49 = stablehlo.add %31, %48 : tensor<128x1x16xf32>
    %50 = stablehlo.broadcast_in_dim %arg13, dims = [1, 2] : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<1x16x512xf32>) -> tensor<128x16x512xf32>
    %52 = stablehlo.multiply %51, %cst_12 : tensor<128x16x512xf32>
    %53 = stablehlo.broadcast_in_dim %arg12, dims = [1, 2] : (tensor<16x512xf32>) -> tensor<1x16x512xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<1x16x512xf32>) -> tensor<128x16x512xf32>
    %55 = stablehlo.add %54, %52 : tensor<128x16x512xf32>
    %56 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %58 = stablehlo.multiply %57, %cst_8 : tensor<128x512xf32>
    %59 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<128x512xf32>
    %61 = stablehlo.add %60, %58 : tensor<128x512xf32>
    %62 = stablehlo.dot_general %49, %55, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x16xf32>, tensor<128x16x512xf32>) -> tensor<128x1x512xf32>
    %63 = stablehlo.broadcast_in_dim %61, dims = [0, 2] : (tensor<128x512xf32>) -> tensor<128x1x512xf32>
    %64 = stablehlo.add %62, %63 : tensor<128x1x512xf32>
    %65 = stablehlo.tanh %64 : tensor<128x1x512xf32>
    %66 = stablehlo.broadcast_in_dim %arg17, dims = [1, 2] : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<1x512x784xf32>) -> tensor<128x512x784xf32>
    %68 = stablehlo.multiply %67, %cst_13 : tensor<128x512x784xf32>
    %69 = stablehlo.broadcast_in_dim %arg16, dims = [1, 2] : (tensor<512x784xf32>) -> tensor<1x512x784xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2] : (tensor<1x512x784xf32>) -> tensor<128x512x784xf32>
    %71 = stablehlo.add %70, %68 : tensor<128x512x784xf32>
    %72 = stablehlo.broadcast_in_dim %arg19, dims = [1] : (tensor<784xf32>) -> tensor<1x784xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<128x784xf32>
    %74 = stablehlo.multiply %73, %cst_14 : tensor<128x784xf32>
    %75 = stablehlo.broadcast_in_dim %arg18, dims = [1] : (tensor<784xf32>) -> tensor<1x784xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1] : (tensor<1x784xf32>) -> tensor<128x784xf32>
    %77 = stablehlo.add %76, %74 : tensor<128x784xf32>
    %78 = stablehlo.dot_general %65, %71, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<128x1x512xf32>, tensor<128x512x784xf32>) -> tensor<128x1x784xf32>
    %79 = stablehlo.broadcast_in_dim %77, dims = [0, 2] : (tensor<128x784xf32>) -> tensor<128x1x784xf32>
    %80 = stablehlo.add %78, %79 : tensor<128x1x784xf32>
    %81 = stablehlo.negate %80 : tensor<128x1x784xf32>
    %82 = stablehlo.exponential %81 : tensor<128x1x784xf32>
    %83 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1x784xf32>
    %84 = stablehlo.add %83, %82 : tensor<128x1x784xf32>
    %85 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1x784xf32>
    %86 = stablehlo.divide %85, %84 : tensor<128x1x784xf32>
    %87 = stablehlo.broadcast_in_dim %arg20, dims = [0, 2] : (tensor<128x784xf32>) -> tensor<128x1x784xf32>
    %88 = stablehlo.subtract %87, %86 : tensor<128x1x784xf32>
    %89 = stablehlo.multiply %88, %88 : tensor<128x1x784xf32>
    %90 = stablehlo.reduce(%89 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<128x1x784xf32>, tensor<f32>) -> tensor<128x1xf32>
    %91 = stablehlo.reduce(%90 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x1xf32>, tensor<f32>) -> tensor<f32>
    %92 = stablehlo.divide %91, %cst_4 : tensor<f32>
    %93 = stablehlo.exponential %46 : tensor<128x1x16xf32>
    %94 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<128x1x16xf32>
    %95 = stablehlo.add %94, %46 : tensor<128x1x16xf32>
    %96 = stablehlo.multiply %31, %31 : tensor<128x1x16xf32>
    %97 = stablehlo.subtract %95, %96 : tensor<128x1x16xf32>
    %98 = stablehlo.subtract %97, %93 : tensor<128x1x16xf32>
    %99 = stablehlo.reduce(%98 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<128x1x16xf32>, tensor<f32>) -> tensor<128x1xf32>
    %100 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<128x1xf32>
    %101 = stablehlo.multiply %100, %99 : tensor<128x1xf32>
    %102 = stablehlo.reduce(%101 init: %cst_5) applies stablehlo.add across dimensions = [0, 1] : (tensor<128x1xf32>, tensor<f32>) -> tensor<f32>
    %103 = stablehlo.divide %102, %cst_4 : tensor<f32>
    %104 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %105 = stablehlo.multiply %arg1, %104 : tensor<784x512xf32>
    %106 = stablehlo.log %105 : tensor<784x512xf32>
    %107 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784x512xf32>
    %108 = stablehlo.add %107, %106 : tensor<784x512xf32>
    %109 = stablehlo.multiply %arg0, %arg0 : tensor<784x512xf32>
    %110 = stablehlo.subtract %108, %109 : tensor<784x512xf32>
    %111 = stablehlo.subtract %110, %105 : tensor<784x512xf32>
    %112 = stablehlo.reduce(%111 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<784x512xf32>, tensor<f32>) -> tensor<784xf32>
    %113 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %114 = stablehlo.multiply %113, %112 : tensor<784xf32>
    %115 = stablehlo.reduce(%114 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<784xf32>, tensor<f32>) -> tensor<f32>
    %116 = stablehlo.divide %115, %cst_1 : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %118 = stablehlo.multiply %arg3, %117 : tensor<512xf32>
    %119 = stablehlo.log %118 : tensor<512xf32>
    %120 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %121 = stablehlo.add %120, %119 : tensor<512xf32>
    %122 = stablehlo.multiply %arg2, %arg2 : tensor<512xf32>
    %123 = stablehlo.subtract %121, %122 : tensor<512xf32>
    %124 = stablehlo.subtract %123, %118 : tensor<512xf32>
    %125 = stablehlo.reduce(%124 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %126 = stablehlo.multiply %cst_3, %125 : tensor<f32>
    %127 = stablehlo.reduce(%126 init: %cst_5) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %128 = stablehlo.divide %127, %cst_6 : tensor<f32>
    %129 = stablehlo.add %116, %128 : tensor<f32>
    %130 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %131 = stablehlo.multiply %arg5, %130 : tensor<512x16xf32>
    %132 = stablehlo.log %131 : tensor<512x16xf32>
    %133 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %134 = stablehlo.add %133, %132 : tensor<512x16xf32>
    %135 = stablehlo.multiply %arg4, %arg4 : tensor<512x16xf32>
    %136 = stablehlo.subtract %134, %135 : tensor<512x16xf32>
    %137 = stablehlo.subtract %136, %131 : tensor<512x16xf32>
    %138 = stablehlo.reduce(%137 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %139 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %140 = stablehlo.multiply %139, %138 : tensor<512xf32>
    %141 = stablehlo.reduce(%140 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %142 = stablehlo.divide %141, %cst_0 : tensor<f32>
    %143 = stablehlo.add %129, %142 : tensor<f32>
    %144 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %145 = stablehlo.multiply %arg7, %144 : tensor<16xf32>
    %146 = stablehlo.log %145 : tensor<16xf32>
    %147 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %148 = stablehlo.add %147, %146 : tensor<16xf32>
    %149 = stablehlo.multiply %arg6, %arg6 : tensor<16xf32>
    %150 = stablehlo.subtract %148, %149 : tensor<16xf32>
    %151 = stablehlo.subtract %150, %145 : tensor<16xf32>
    %152 = stablehlo.reduce(%151 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %153 = stablehlo.multiply %cst_3, %152 : tensor<f32>
    %154 = stablehlo.reduce(%153 init: %cst_5) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %155 = stablehlo.divide %154, %cst_6 : tensor<f32>
    %156 = stablehlo.add %143, %155 : tensor<f32>
    %157 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %158 = stablehlo.multiply %arg9, %157 : tensor<512x16xf32>
    %159 = stablehlo.log %158 : tensor<512x16xf32>
    %160 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x16xf32>
    %161 = stablehlo.add %160, %159 : tensor<512x16xf32>
    %162 = stablehlo.multiply %arg8, %arg8 : tensor<512x16xf32>
    %163 = stablehlo.subtract %161, %162 : tensor<512x16xf32>
    %164 = stablehlo.subtract %163, %158 : tensor<512x16xf32>
    %165 = stablehlo.reduce(%164 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<512x16xf32>, tensor<f32>) -> tensor<512xf32>
    %166 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %167 = stablehlo.multiply %166, %165 : tensor<512xf32>
    %168 = stablehlo.reduce(%167 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %169 = stablehlo.divide %168, %cst_0 : tensor<f32>
    %170 = stablehlo.add %156, %169 : tensor<f32>
    %171 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %172 = stablehlo.multiply %arg11, %171 : tensor<16xf32>
    %173 = stablehlo.log %172 : tensor<16xf32>
    %174 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %175 = stablehlo.add %174, %173 : tensor<16xf32>
    %176 = stablehlo.multiply %arg10, %arg10 : tensor<16xf32>
    %177 = stablehlo.subtract %175, %176 : tensor<16xf32>
    %178 = stablehlo.subtract %177, %172 : tensor<16xf32>
    %179 = stablehlo.reduce(%178 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %180 = stablehlo.multiply %cst_3, %179 : tensor<f32>
    %181 = stablehlo.reduce(%180 init: %cst_5) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %182 = stablehlo.divide %181, %cst_6 : tensor<f32>
    %183 = stablehlo.add %170, %182 : tensor<f32>
    %184 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %185 = stablehlo.multiply %arg13, %184 : tensor<16x512xf32>
    %186 = stablehlo.log %185 : tensor<16x512xf32>
    %187 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<16x512xf32>
    %188 = stablehlo.add %187, %186 : tensor<16x512xf32>
    %189 = stablehlo.multiply %arg12, %arg12 : tensor<16x512xf32>
    %190 = stablehlo.subtract %188, %189 : tensor<16x512xf32>
    %191 = stablehlo.subtract %190, %185 : tensor<16x512xf32>
    %192 = stablehlo.reduce(%191 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<16x512xf32>, tensor<f32>) -> tensor<16xf32>
    %193 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %194 = stablehlo.multiply %193, %192 : tensor<16xf32>
    %195 = stablehlo.reduce(%194 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %196 = stablehlo.divide %195, %cst : tensor<f32>
    %197 = stablehlo.add %183, %196 : tensor<f32>
    %198 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %199 = stablehlo.multiply %arg15, %198 : tensor<512xf32>
    %200 = stablehlo.log %199 : tensor<512xf32>
    %201 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %202 = stablehlo.add %201, %200 : tensor<512xf32>
    %203 = stablehlo.multiply %arg14, %arg14 : tensor<512xf32>
    %204 = stablehlo.subtract %202, %203 : tensor<512xf32>
    %205 = stablehlo.subtract %204, %199 : tensor<512xf32>
    %206 = stablehlo.reduce(%205 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %207 = stablehlo.multiply %cst_3, %206 : tensor<f32>
    %208 = stablehlo.reduce(%207 init: %cst_5) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %209 = stablehlo.divide %208, %cst_6 : tensor<f32>
    %210 = stablehlo.add %197, %209 : tensor<f32>
    %211 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %212 = stablehlo.multiply %arg17, %211 : tensor<512x784xf32>
    %213 = stablehlo.log %212 : tensor<512x784xf32>
    %214 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<512x784xf32>
    %215 = stablehlo.add %214, %213 : tensor<512x784xf32>
    %216 = stablehlo.multiply %arg16, %arg16 : tensor<512x784xf32>
    %217 = stablehlo.subtract %215, %216 : tensor<512x784xf32>
    %218 = stablehlo.subtract %217, %212 : tensor<512x784xf32>
    %219 = stablehlo.reduce(%218 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<512x784xf32>, tensor<f32>) -> tensor<512xf32>
    %220 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %221 = stablehlo.multiply %220, %219 : tensor<512xf32>
    %222 = stablehlo.reduce(%221 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %223 = stablehlo.divide %222, %cst_0 : tensor<f32>
    %224 = stablehlo.add %210, %223 : tensor<f32>
    %225 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %226 = stablehlo.multiply %arg19, %225 : tensor<784xf32>
    %227 = stablehlo.log %226 : tensor<784xf32>
    %228 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<784xf32>
    %229 = stablehlo.add %228, %227 : tensor<784xf32>
    %230 = stablehlo.multiply %arg18, %arg18 : tensor<784xf32>
    %231 = stablehlo.subtract %229, %230 : tensor<784xf32>
    %232 = stablehlo.subtract %231, %226 : tensor<784xf32>
    %233 = stablehlo.reduce(%232 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<784xf32>, tensor<f32>) -> tensor<f32>
    %234 = stablehlo.multiply %cst_3, %233 : tensor<f32>
    %235 = stablehlo.reduce(%234 init: %cst_5) applies stablehlo.add across dimensions = [] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %236 = stablehlo.divide %235, %cst_6 : tensor<f32>
    %237 = stablehlo.add %224, %236 : tensor<f32>
    %238 = stablehlo.add %92, %103 : tensor<f32>
    %239 = stablehlo.add %238, %237 : tensor<f32>
    return %239 : tensor<f32>
  }
}
