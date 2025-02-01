let fn = Ir.create_func

module Var = Ir.Var

let matmul a b =
  let a_shape = Ir.shape_of_var a in
  let b_shape = Ir.shape_of_var b in
  let rec prefix = function
    | [_; _] ->
        []
    | x :: xs ->
        x :: prefix xs
    | [] ->
        failwith "matmul requires at least 2 dims"
  in
  let rec common_prefix a b =
    match (a, b) with
    | [], _ | _, [] ->
        []
    | a :: a_tl, b :: b_tl when a = b ->
        a :: common_prefix a_tl b_tl
    | _ ->
        []
  in
  let batching_dims =
    List.init
      (List.length (common_prefix (prefix a_shape) (prefix b_shape)))
      Fun.id
  in
  Ir.Var.DotProduct
    ( a
    , b
    , [List.length a_shape - 1]
    , [List.length b_shape - 2]
    , batching_dims
    , batching_dims )

let ( +@ ) a b = Var.Add (a, b)

let ( -@ ) a b = Var.Subtract (a, b)

let ( *@ ) a b = Var.Multiply (a, b)

let ( /@ ) a b = Var.Divide (a, b)

let ( <<@ ) a b = Var.LeftShift (a, b)

let ( >>@ ) a b = Var.RightShift (a, b)

let ( |@ ) a b = Var.Or (a, b)

let exp a = Var.Exponential a

let pow a b = Var.Pow (a, b)

let ( **@ ) = pow

let abs a = Var.Abs a

let ln a = Var.Ln a

let compare dir a b = Var.Compare (a, dir, b)

let min a b = Var.Min (a, b)

let max a b = Var.Max (a, b)

let ( =@ ) a = compare Ir.Eq a

let ( <>@ ) a = compare Ir.Ne a

let ( >=@ ) a = compare Ir.Ge a

let ( >@ ) a = compare Ir.Gt a

let ( <=@ ) a = compare Ir.Le a

let ( <@ ) a = compare Ir.Lt a

let broadcast_scalar op shape = Ir.Var.BroadcastInDim (op, shape)

let broadcast_scalar_like op var = broadcast_scalar op (Ir.shape_of_var var)

let tensor_to_ir tensor = Ir.Var.Constant tensor

let full kind value shape = Ir.Var.BroadcastScalarConstant ((shape, kind), value)

let full_f32 = full F32

let full_i1 = full I1

let full_like kind value var = Ir.shape_of_var var |> full kind value

let var_float_op op a b = op a (full_like F32 b a)

let float_var_op op a b = op (full_like F32 a b) b

let var_u64_op op a b = op a (full_like U64 b a)

let u64_var_op op a b = op (full_like U64 a b) b

let ( +.> ) = var_float_op ( +@ )

let ( -.> ) = var_float_op ( -@ )

let ( *.> ) = var_float_op ( *@ )

let ( /.> ) = var_float_op ( /@ )

let ( **.> ) = var_float_op ( **@ )

let ( =.> ) = var_float_op ( =@ )

let ( <>.> ) = var_float_op ( <>@ )

let ( >=.> ) = var_float_op ( >=@ )

let ( >.> ) = var_float_op ( >@ )

let ( <=.> ) = var_float_op ( <=@ )

let ( <.> ) = var_float_op ( <@ )

let ( <<.> ) = var_u64_op ( <<@ )

let ( >>.> ) = var_u64_op ( >>@ )

let ( |.> ) = var_u64_op ( |@ )

let ( +.< ) = float_var_op ( +@ )

let ( -.< ) = float_var_op ( -@ )

let ( *.< ) = float_var_op ( *@ )

let ( /.< ) = float_var_op ( /@ )

let ( **.< ) = float_var_op ( **@ )

let ( =.< ) = float_var_op ( =@ )

let ( <>.< ) = float_var_op ( <>@ )

let ( >=.< ) = float_var_op ( >=@ )

let ( >.< ) = float_var_op ( >@ )

let ( <=.< ) = float_var_op ( <=@ )

let ( <.< ) = float_var_op ( <@ )

let ( <<.< ) = u64_var_op ( <<@ )

let ( >>.< ) = u64_var_op ( >>@ )

let ( |.< ) = u64_var_op ( |@ )

let sqrt a = a **.> 0.5

let ( ~-@ ) a = Var.Negate a

let tanh a = Var.Tanh a

let ones : type a b. (a, b) Ir.ValueType.u -> (a, b) Ir.Var.u = function
  | shape, F32 ->
      full F32 1. shape
  | shape, F64 ->
      full F64 1. shape
  | shape, I1 ->
      full I1 false shape
  | shape, I64 ->
      full I64 Signed.Int64.one shape
  | shape, U32 ->
      full U32 Unsigned.UInt32.one shape
  | shape, U64 ->
      full U64 Unsigned.UInt64.one shape

let ones_like t = ones (Ir.ValueType.of_var t)

let zeros : type a b. (a, b) Ir.ValueType.u -> (a, b) Ir.Var.u = function
  | shape, F32 ->
      full F32 0. shape
  | shape, F64 ->
      full F64 0. shape
  | shape, I1 ->
      full I1 false shape
  | shape, I64 ->
      full I64 Signed.Int64.zero shape
  | shape, U32 ->
      full U32 Unsigned.UInt32.zero shape
  | shape, U64 ->
      full U64 Unsigned.UInt64.zero shape

let zeros_like t = zeros (Ir.ValueType.of_var t)

let norm mean stddev shape =
  Var.Random
    ( (shape, F32)
    , mean
    , stddev
    , List.map Signed.Int64.of_int shape
      |> Ir.Tensor.of_list I64 [List.length shape]
      |> tensor_to_ir
    , Normal )

let uniform low high shape =
  Var.Random
    ( (shape, F32)
    , low
    , high
    , List.map Signed.Int64.of_int shape
      |> Ir.Tensor.of_list I64 [List.length shape]
      |> tensor_to_ir
    , Uniform )

let sum axes x = Var.Sum (x, axes)

let mean axes x =
  let shape, _ = Ir.ValueType.of_var x in
  let size =
    List.filteri (fun i _ -> List.mem i axes) shape |> List.fold_left ( * ) 1
  in
  sum axes x /.> float_of_int size

let transpose var permutation =
  let shape = Ir.shape_of_var var in
  if Stdlib.(List.length permutation <> List.length shape) then
    failwith "Permutation length must match tensor rank" ;
  if
    not
      Stdlib.(
        List.sort compare permutation = List.init (List.length shape) Fun.id )
  then failwith "Invalid permutation" ;
  Var.Transpose (var, permutation)

let scalar_f32 x = full F32 x []

let scalar_u64 str = full U64 (Unsigned.UInt64.of_string str) []

let assert_float_fn
    (f : (Ir.Tensor.f32, float) Ir.Var.u -> (Ir.Tensor.f32, float) Ir.Var.u) :
    Ir.Var.map_fn =
  let f : type a b. (a, b) Ir.Var.u -> (a, b) Ir.Var.u =
   fun x ->
    match Ir.ValueType.of_var x with
    | _, F32 ->
        f x
    | _ ->
        failwith "assert_float_map: unsupported type"
  in
  {f}

let assert_float2_fn
    (f :
         (Ir.Tensor.f32, float) Ir.Var.u
      -> (Ir.Tensor.f32, float) Ir.Var.u
      -> (Ir.Tensor.f32, float) Ir.Var.u ) : Ir.Var.map2_fn =
  let f : type a b. (a, b) Ir.Var.u -> (a, b) Ir.Var.u -> (a, b) Ir.Var.u =
   fun x y ->
    match Ir.ValueType.of_var x with
    | _, F32 ->
        f x y
    | _ ->
        failwith "assert_float_map: unsupported type"
  in
  {f}

let float_map f = Ir.Var.map (assert_float_fn f)

let float_map2 f = Ir.Var.map2 (assert_float2_fn f)

let bitcast dtype var = Var.Bitcast (var, dtype)

let convert dtype var = Var.Convert (var, dtype)

let iota n var = Var.Iota (n, var)

let reshape shape var = Var.Reshape (var, shape)

let no_grad x = Ir.Var.NoGrad x

let sin x = Var.Sin x

let cos x = Var.Cos x

let concat axis vars = Var.Concatenate (vars, axis)
