let fn = Translation.create_func

let matmul a b =
  let a_shape = Var.shape a in
  let b_shape = Var.shape b in
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
  Var.DotProduct
    ( a
    , b
    , [List.length a_shape - 1]
    , [List.length b_shape - 2]
    , batching_dims
    , batching_dims )

let ( @$ ) a b = matmul a b

let%expect_test "matmul_operator" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a @$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| dot(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let assert_same_shape a b =
  let a_shape = Var.shape a in
  let b_shape = Var.shape b in
  if List.length a_shape <> List.length b_shape then
    failwith "assert_same_shape: different number of dimensions" ;
  if not (List.for_all2 ( = ) a_shape b_shape) then
    failwith "assert_same_shape: different shapes"

let ( +$ ) a b =
  assert_same_shape a b ;
  Var.Add (a, b)

let%expect_test "add" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = a +$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) + const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])) |}]

let ( -$ ) a b =
  assert_same_shape a b ;
  Var.Subtract (a, b)

let%expect_test "subtract" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = a -$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) - const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])) |}]

let ( *$ ) : type a b. (a, b) Var.u -> (a, b) Var.u -> (a, b) Var.u =
 fun a b ->
  assert_same_shape a b ;
  match Var.value_type a with
  | _, (Tensor.I64 | Tensor.U64) when !Metal.enabled && a = b ->
      (* The Metal PJRT plugin rewrites multiplications to a square instruction if the second argument is the same as the first argument.
       However, there are no integer square instructions. *)
      Var.Multiply (Var.OptimizationBarrier a, b)
  | _ ->
      Var.Multiply (a, b)

let%expect_test "multiply" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = a *$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) * const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])) |}]

let ( /$ ) a b =
  assert_same_shape a b ;
  Var.Divide (a, b)

let%expect_test "divide" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = a /$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) / const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])) |}]

let ( <<$ ) a b =
  assert_same_shape a b ;
  Var.LeftShift (a, b)

let%expect_test "left_shift" =
  let a =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let b =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.one
         ; Unsigned.UInt64.one
         ; Unsigned.UInt64.one ] )
  in
  let result = a <<$ b in
  print_endline (Var.to_string result) ;
  [%expect {| (const([[1, 2], [3, 4]]) << const([[1, 1], [1, 1]])) |}]

let ( >>$ ) a b =
  assert_same_shape a b ;
  Var.RightShift (a, b)

let%expect_test "right_shift" =
  let a =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.of_int 4
         ; Unsigned.UInt64.of_int 8
         ; Unsigned.UInt64.of_int 16
         ; Unsigned.UInt64.of_int 32 ] )
  in
  let b =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.one
         ; Unsigned.UInt64.one
         ; Unsigned.UInt64.one ] )
  in
  let result = a >>$ b in
  print_endline (Var.to_string result) ;
  [%expect {| (const([[4, 8], [16, 32]]) >> const([[1, 1], [1, 1]])) |}]

let ( |$ ) a b =
  assert_same_shape a b ;
  Var.Or (a, b)

let%expect_test "or" =
  let a =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let b =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.of_int 5
         ; Unsigned.UInt64.of_int 6
         ; Unsigned.UInt64.of_int 7
         ; Unsigned.UInt64.of_int 8 ] )
  in
  let result = a |$ b in
  print_endline (Var.to_string result) ;
  [%expect {| (const([[1, 2], [3, 4]]) | const([[5, 6], [7, 8]])) |}]

let exp a = Var.Exponential a

let%expect_test "exp" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = exp a in
  print_endline (Var.to_string result) ;
  [%expect
    {| exp(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let pow a b =
  assert_same_shape a b ;
  Var.Pow (a, b)

let%expect_test "pow" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [2.0; 2.0; 2.0; 2.0]) in
  let result = pow a b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) ^ const([[2.000000e+00, 2.000000e+00], [2.000000e+00, 2.000000e+00]])) |}]

let ( **$ ) = pow

let abs a = Var.Abs a

let%expect_test "abs" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [-1.0; -2.0; 3.0; -4.0]) in
  let result = abs a in
  print_endline (Var.to_string result) ;
  [%expect
    {| |const([[-1.000000e+00, -2.000000e+00], [3.000000e+00, -4.000000e+00]])| |}]

let ln a = Var.Ln a

let%expect_test "ln" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = ln a in
  print_endline (Var.to_string result) ;
  [%expect
    {| ln(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ln1p a = Var.Ln_1_plus a

let%expect_test "ln1p" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = ln1p a in
  print_endline (Var.to_string result) ;
  [%expect
    {| ln(1 + const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let compare dir a b =
  assert_same_shape a b ;
  Var.Compare (a, dir, b)

let%expect_test "compare" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 3.0; 2.0; 4.0]) in
  let result = compare Var.Eq a b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) == const([[1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]])) |}]

let min a b =
  assert_same_shape a b ;
  Var.Min (a, b)

let%expect_test "min" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [0.0; 3.0; 2.0; 5.0]) in
  let result = min a b in
  print_endline (Var.to_string result) ;
  [%expect
    {| min(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), const([[0.000000e+00, 3.000000e+00], [2.000000e+00, 5.000000e+00]])) |}]

let max a b =
  assert_same_shape a b ;
  Var.Max (a, b)

let%expect_test "max" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [0.0; 3.0; 2.0; 5.0]) in
  let result = max a b in
  print_endline (Var.to_string result) ;
  [%expect
    {| max(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), const([[0.000000e+00, 3.000000e+00], [2.000000e+00, 5.000000e+00]])) |}]

let ( =$ ) a = compare Var.Eq a

let%expect_test "eq" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a =$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) == const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( <>$ ) a = compare Var.Ne a

let%expect_test "ne" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 5.0]) in
  let result = a <>$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) != const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 5.000000e+00]])) |}]

let ( >=$ ) a = compare Var.Ge a

let%expect_test "ge" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 1.0; 4.0; 3.0]) in
  let result = a >=$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) >= const([[1.000000e+00, 1.000000e+00], [4.000000e+00, 3.000000e+00]])) |}]

let ( >$ ) a = compare Var.Gt a

let%expect_test "gt" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [0.0; 1.0; 2.0; 3.0]) in
  let result = a >$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) > const([[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]])) |}]

let ( <=$ ) a = compare Var.Le a

let%expect_test "le" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 3.0; 3.0; 5.0]) in
  let result = a <=$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) <= const([[1.000000e+00, 3.000000e+00], [3.000000e+00, 5.000000e+00]])) |}]

let ( <$ ) a = compare Var.Lt a

let%expect_test "lt" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [2.0; 3.0; 4.0; 5.0]) in
  let result = a <$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) < const([[2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00]])) |}]

let broadcast_scalar op shape = Var.BroadcastInDim (op, shape)

let%expect_test "broadcast_scalar" =
  let op = Var.Constant (Tensor.of_list F32 [] [1.0]) in
  let result = broadcast_scalar op [2; 2] in
  print_endline (Var.to_string result) ;
  [%expect {| broadcast(const(1.000000e+00), 2,2) |}]

let broadcast_scalar_like op var = broadcast_scalar op (Var.shape var)

let%expect_test "broadcast_scalar_like" =
  let op = Var.Constant (Tensor.of_list F32 [] [1.0]) in
  let var = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = broadcast_scalar_like op var in
  print_endline (Var.to_string result) ;
  [%expect {| broadcast(const(1.000000e+00), 2,2) |}]

let tensor_to_ir tensor = Var.Constant tensor

let%expect_test "tensor_to_ir" =
  let tensor = Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0] in
  let result = tensor_to_ir tensor in
  print_endline (Var.to_string result) ;
  [%expect
    {| const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) |}]

let full kind value shape = Var.BroadcastScalarConstant ((shape, kind), value)

let%expect_test "full" =
  let result = full F32 1.0 [2; 2] in
  print_endline (Var.to_string result) ;
  [%expect {| const(1.000000e+00) |}]

let full_f32 = full F32

let%expect_test "full_f32" =
  let result = full_f32 1.0 [2; 2] in
  print_endline (Var.to_string result) ;
  [%expect {| const(1.000000e+00) |}]

let full_i1 = full I1

let%expect_test "full_i1" =
  let result = full_i1 true [2; 2] in
  print_endline (Var.to_string result) ;
  [%expect {| const(true) |}]

let full_like kind value var = Var.shape var |> full kind value

let%expect_test "full_like" =
  let var = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = full_like F32 1.0 var in
  print_endline (Var.to_string result) ;
  [%expect {| const(1.000000e+00) |}]

let var_float_op op a b = op a (full_like F32 b a)

let%expect_test "var_float_op" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = var_float_op ( +$ ) a 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) + const(2.000000e+00)) |}]

let float_var_op op a b = op (full_like F32 a b) b

let%expect_test "float_var_op" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = float_var_op ( +$ ) 2.0 b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) + const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let var_u64_op op a b = op a (full_like U64 b a)

let%expect_test "var_u64_op" =
  let a =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let result = var_u64_op ( <<$ ) a (Unsigned.UInt64.of_int 1) in
  print_endline (Var.to_string result) ;
  [%expect {| (const([[1, 2], [3, 4]]) << const(1)) |}]

let u64_var_op op a b = op (full_like U64 a b) b

let%expect_test "u64_var_op" =
  let b =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let result = u64_var_op ( <<$ ) (Unsigned.UInt64.of_int 1) b in
  print_endline (Var.to_string result) ;
  [%expect {| (const(1) << const([[1, 2], [3, 4]])) |}]

let ( +$. ) = var_float_op ( +$ )

let%expect_test "add_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a +$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) + const(2.000000e+00)) |}]

let ( -$. ) = var_float_op ( -$ )

let%expect_test "subtract_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a -$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) - const(2.000000e+00)) |}]

let ( *$. ) = var_float_op ( *$ )

let%expect_test "multiply_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a *$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) * const(2.000000e+00)) |}]

let ( /$. ) = var_float_op ( /$ )

let%expect_test "divide_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a /$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) / const(2.000000e+00)) |}]

let ( **$. ) = var_float_op ( **$ )

let%expect_test "pow_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a **$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) ^ const(2.000000e+00)) |}]

let ( =$. ) = var_float_op ( =$ )

let%expect_test "eq_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a =$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) == const(2.000000e+00)) |}]

let ( <>.$ ) = var_float_op ( <>$ )

let%expect_test "ne_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a <>.$ 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) != const(2.000000e+00)) |}]

let ( >=$. ) = var_float_op ( >=$ )

let%expect_test "ge_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a >=$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) >= const(2.000000e+00)) |}]

let ( >$. ) = var_float_op ( >$ )

let%expect_test "gt_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a >$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) > const(2.000000e+00)) |}]

let ( <=$. ) = var_float_op ( <=$ )

let%expect_test "le_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a <=$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) <= const(2.000000e+00)) |}]

let ( <$. ) = var_float_op ( <$ )

let%expect_test "lt_scalar_right" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = a <$. 2.0 in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) < const(2.000000e+00)) |}]

let ( <<$. ) = var_u64_op ( <<$ )

let%expect_test "left_shift_scalar_right" =
  let a =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let result = a <<$. Unsigned.UInt64.of_int 1 in
  print_endline (Var.to_string result) ;
  [%expect {| (const([[1, 2], [3, 4]]) << const(1)) |}]

let ( >>$. ) = var_u64_op ( >>$ )

let%expect_test "right_shift_scalar_right" =
  let a =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.of_int 4
         ; Unsigned.UInt64.of_int 8
         ; Unsigned.UInt64.of_int 16
         ; Unsigned.UInt64.of_int 32 ] )
  in
  let result = a >>$. Unsigned.UInt64.of_int 1 in
  print_endline (Var.to_string result) ;
  [%expect {| (const([[4, 8], [16, 32]]) >> const(1)) |}]

let ( |$. ) = var_u64_op ( |$ )

let%expect_test "or_scalar_right" =
  let a =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let result = a |$. Unsigned.UInt64.of_int 1 in
  print_endline (Var.to_string result) ;
  [%expect {| (const([[1, 2], [3, 4]]) | const(1)) |}]

let ( +.$ ) = float_var_op ( +$ )

let%expect_test "add_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 +.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) + const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( -.$ ) = float_var_op ( -$ )

let%expect_test "subtract_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 -.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) - const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( *.$ ) = float_var_op ( *$ )

let%expect_test "multiply_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 *.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) * const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( /.$ ) = float_var_op ( /$ )

let%expect_test "divide_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 /.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) / const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( **.$ ) = float_var_op ( **$ )

let%expect_test "pow_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 **.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) ^ const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( =.$ ) = float_var_op ( =$ )

let%expect_test "eq_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 =.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) == const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( <>.$ ) = float_var_op ( <>$ )

let%expect_test "ne_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 <>.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) != const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( >=.$ ) = float_var_op ( >=$ )

let%expect_test "ge_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 >=.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) >= const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( >.$ ) = float_var_op ( >$ )

let%expect_test "gt_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 >.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) > const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( <=.$ ) = float_var_op ( <=$ )

let%expect_test "le_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 <=.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) <= const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( <.$ ) = float_var_op ( <$ )

let%expect_test "lt_scalar_left" =
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = 2.0 <.$ b in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const(2.000000e+00) < const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ( <<.$ ) = u64_var_op ( <<$ )

let%expect_test "left_shift_scalar_left" =
  let b =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let result = Unsigned.UInt64.of_int 1 <<.$ b in
  print_endline (Var.to_string result) ;
  [%expect {| (const(1) << const([[1, 2], [3, 4]])) |}]

let ( >>.$ ) = u64_var_op ( >>$ )

let%expect_test "right_shift_scalar_left" =
  let b =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.of_int 4
         ; Unsigned.UInt64.of_int 8
         ; Unsigned.UInt64.of_int 16
         ; Unsigned.UInt64.of_int 32 ] )
  in
  let result = Unsigned.UInt64.of_int 1 >>.$ b in
  print_endline (Var.to_string result) ;
  [%expect {| (const(1) >> const([[4, 8], [16, 32]])) |}]

let ( |.$ ) = u64_var_op ( |$ )

let%expect_test "or_scalar_left" =
  let b =
    Var.Constant
      (Tensor.of_list U64 [2; 2]
         [ Unsigned.UInt64.one
         ; Unsigned.UInt64.of_int 2
         ; Unsigned.UInt64.of_int 3
         ; Unsigned.UInt64.of_int 4 ] )
  in
  let result = Unsigned.UInt64.of_int 1 |.$ b in
  print_endline (Var.to_string result) ;
  [%expect {| (const(1) | const([[1, 2], [3, 4]])) |}]

let sqrt a = Var.Sqrt a

let%expect_test "sqrt" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 4.0; 9.0; 16.0]) in
  let result = sqrt a in
  print_endline (Var.to_string result) ;
  [%expect
    {| sqrt(const([[1.000000e+00, 4.000000e+00], [9.000000e+00, 1.600000e+01]])) |}]

let ( ~-$ ) a = Var.Negate a

let%expect_test "negate" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = ~-$a in
  print_endline (Var.to_string result) ;
  [%expect
    {| (-const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let tanh a = Var.Tanh a

let%expect_test "tanh" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = tanh a in
  print_endline (Var.to_string result) ;
  [%expect
    {| tanh(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let ones : type a b. (a, b) Value_type.u -> (a, b) Var.u = function
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

let%expect_test "ones" =
  let result = ones ([2; 2], F32) in
  print_endline (Var.to_string result) ;
  [%expect {| const(1.000000e+00) |}]

let ones_like t = ones (Var.value_type t)

let%expect_test "ones_like" =
  let t = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = ones_like t in
  print_endline (Var.to_string result) ;
  [%expect {| const(1.000000e+00) |}]

let zeros : type a b. (a, b) Value_type.u -> (a, b) Var.u = function
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

let%expect_test "zeros" =
  let result = zeros ([2; 2], F32) in
  print_endline (Var.to_string result) ;
  [%expect {| const(0.000000e+00) |}]

let zeros_like t = zeros (Var.value_type t)

let%expect_test "zeros_like" =
  let t = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = zeros_like t in
  print_endline (Var.to_string result) ;
  [%expect {| const(0.000000e+00) |}]

let norm mean stddev shape =
  Var.Random
    ( (shape, F32)
    , mean
    , stddev
    , List.map Signed.Int64.of_int shape
      |> Tensor.of_list I64 [List.length shape]
      |> tensor_to_ir
    , Normal )

let%expect_test "norm" =
  let mean = full F32 0.0 [] in
  let stddev = full F32 1.0 [] in
  let result = norm mean stddev [2; 2] in
  print_endline (Var.to_string result) ;
  [%expect
    {| random(const(0.000000e+00), const(1.000000e+00), const([2, 2]), normal) |}]

let uniform low high shape =
  Var.Random
    ( (shape, F32)
    , low
    , high
    , List.map Signed.Int64.of_int shape
      |> Tensor.of_list I64 [List.length shape]
      |> tensor_to_ir
    , Uniform )

let%expect_test "uniform" =
  let low = full F32 0.0 [] in
  let high = full F32 1.0 [] in
  let result = uniform low high [2; 2] in
  print_endline (Var.to_string result) ;
  [%expect
    {| random(const(0.000000e+00), const(1.000000e+00), const([2, 2]), uniform) |}]

let sum ?axes x =
  let shape = Var.shape x in
  let axes =
    match axes with
    | Some axes ->
        axes
    | None ->
        List.init (List.length shape) Fun.id
  in
  Var.Sum (x, axes)

let%expect_test "sum" =
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = sum ~axes:[0] x in
  print_endline (Var.to_string result) ;
  [%expect
    {| sum(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), 0) |}]

let%expect_test "sum_all_axes" =
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = sum x in
  print_endline (Var.to_string result) ;
  [%expect
    {| sum(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), 0,1) |}]

let mean ?axes x =
  let shape = Var.shape x in
  let axes =
    match axes with
    | Some axes ->
        axes
    | None ->
        List.init (List.length shape) Fun.id
  in
  let size =
    List.filteri (fun i _ -> List.mem i axes) shape |> List.fold_left ( * ) 1
  in
  sum ~axes (x /$. float_of_int size)

let%expect_test "mean" =
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = mean ~axes:[0] x in
  print_endline (Var.to_string result) ;
  [%expect
    {| sum((const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) / const(2.000000e+00)), 0) |}]

let%expect_test "mean_all_axes" =
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = mean x in
  print_endline (Var.to_string result) ;
  [%expect
    {| sum((const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) / const(4.000000e+00)), 0,1) |}]

let transpose var permutation =
  let shape = Var.shape var in
  if Stdlib.(List.length permutation <> List.length shape) then
    failwith "Permutation length must match tensor rank" ;
  if
    not
      Stdlib.(
        List.sort compare permutation = List.init (List.length shape) Fun.id )
  then failwith "Invalid permutation" ;
  Var.Transpose (var, permutation)

let%expect_test "transpose" =
  let var = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = transpose var [1; 0] in
  print_endline (Var.to_string result) ;
  [%expect
    {| transpose(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), 1,0) |}]

let scalar_f32 x = full F32 x []

let%expect_test "scalar_f32" =
  let result = scalar_f32 1.0 in
  print_endline (Var.to_string result) ;
  [%expect {| const(1.000000e+00) |}]

let ( ~. ) = scalar_f32

let%expect_test "scalar_float" =
  let result = ~.1.0 in
  print_endline (Var.to_string result) ;
  [%expect {| const(1.000000e+00) |}]

let scalar_u64 str = full U64 (Unsigned.UInt64.of_string str) []

let%expect_test "scalar_u64" =
  let result = scalar_u64 "1" in
  print_endline (Var.to_string result) ;
  [%expect {| const(1) |}]

let assert_float_fn (f : (Tensor.f32, float) Var.u -> (Tensor.f32, float) Var.u)
    : Var.map_fn =
  let f : type a b. (a, b) Var.u -> (a, b) Var.u =
   fun x ->
    match Var.value_type x with
    | _, F32 ->
        f x
    | _ ->
        failwith "assert_float_map: unsupported type"
  in
  {f}

let%expect_test "assert_float_fn" =
  let f x = x +$ x in
  let fn = assert_float_fn f in
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = fn.f x in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) + const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let assert_float2_fn
    (f :
         (Tensor.f32, float) Var.u
      -> (Tensor.f32, float) Var.u
      -> (Tensor.f32, float) Var.u ) : Var.map2_fn =
  let f : type a b. (a, b) Var.u -> (a, b) Var.u -> (a, b) Var.u =
   fun x y ->
    match Var.value_type x with
    | _, F32 ->
        f x y
    | _ ->
        failwith "assert_float_map: unsupported type"
  in
  {f}

let%expect_test "assert_float2_fn" =
  let f x y = x +$ y in
  let fn = assert_float2_fn f in
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let y = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = fn.f x y in
  print_endline (Var.to_string result) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) + const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])) |}]

let float_map f = Var.map (assert_float_fn f)

let%expect_test "float_map" =
  let f x = x +$ x in
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = float_map f (E x) in
  print_endline (Var.to_string (Var.List.unwrap result)) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) + const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let float_map2 f = Var.map2 (assert_float2_fn f)

let%expect_test "float_map2" =
  let f x y = x +$ y in
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let y = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = float_map2 f (E x) (E y) in
  print_endline (Var.to_string (Var.List.unwrap result)) ;
  [%expect
    {| (const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]) + const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])) |}]

let bitcast dtype var = Var.Bitcast (var, dtype)

let%expect_test "bitcast" =
  let var = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = bitcast I64 var in
  print_endline (Var.to_string result) ;
  [%expect
    {| bitcast(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let convert dtype var = Var.Convert (var, dtype)

let%expect_test "convert" =
  let var = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = convert I64 var in
  print_endline (Var.to_string result) ;
  [%expect
    {| convert(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let iota n var = Var.Iota (n, var)

let%expect_test "iota" =
  let result = iota 0 [2; 2] in
  print_endline (Var.to_string result) ;
  [%expect {| iota(0, 2,2) |}]

let reshape shape var =
  let size1 = List.fold_left ( * ) 1 shape in
  let size2 = List.fold_left ( * ) 1 (Var.shape var) in
  if size1 <> size2 then
    failwith
      (Printf.sprintf "reshape: incompatible shapes %s and %s"
         (String.concat ", " (List.map string_of_int shape))
         (String.concat ", " (List.map string_of_int (Var.shape var))) ) ;
  Var.Reshape (var, shape)

let%expect_test "reshape" =
  let var = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = reshape [4] var in
  print_endline (Var.to_string result) ;
  [%expect
    {| reshape(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), 4) |}]

let no_grad x = Var.NoGrad x

let%expect_test "no_grad" =
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = no_grad x in
  print_endline (Var.to_string result) ;
  [%expect
    {| nograd(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let sin x = Var.Sin x

let%expect_test "sin" =
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = sin x in
  print_endline (Var.to_string result) ;
  [%expect
    {| sin(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let cos x = Var.Cos x

let%expect_test "cos" =
  let x = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let result = cos x in
  print_endline (Var.to_string result) ;
  [%expect
    {| cos(const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]])) |}]

let concat axis vars = Var.Concatenate (vars, axis)

let%expect_test "concat" =
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = concat 0 [a; b] in
  print_endline (Var.to_string result) ;
  [%expect
    {| concat([const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]); const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])], 0) |}]

let select cond a b =
  assert_same_shape a b ;
  assert_same_shape cond a ;
  Var.Select (cond, a, b)

let%expect_test "select" =
  let cond =
    Var.Constant (Tensor.of_list I1 [2; 2] [true; false; true; false])
  in
  let a = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  let b = Var.Constant (Tensor.of_list F32 [2; 2] [5.0; 6.0; 7.0; 8.0]) in
  let result = select cond a b in
  print_endline (Var.to_string result) ;
  [%expect
    {| select(const([[true, false], [true, false]]), const([[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]), const([[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]])) |}]

let print_shape var =
  let shape = Var.shape var in
  let shape = List.map string_of_int shape in
  Printf.printf "Shape: [%s]\n" (String.concat "; " shape)

let%expect_test "print_shape" =
  let var = Var.Constant (Tensor.of_list F32 [2; 2] [1.0; 2.0; 3.0; 4.0]) in
  print_shape var ; [%expect {|Shape: [2; 2]|}]
