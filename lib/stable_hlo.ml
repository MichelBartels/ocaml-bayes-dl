type tensor_element_type = F32 | F64 | I1 | I64 | U32 | U64

let tensor_element_type_to_string = function
  | F32 ->
      "f32"
  | F64 ->
      "f64"
  | I1 ->
      "i1"
  | I64 ->
      "i64"
  | U32 ->
      "ui32"
  | U64 ->
      "ui64"

type shape = int list

let shape_to_string shape =
  String.concat "" (List.map (fun dim -> string_of_int dim ^ "x") shape)

let%expect_test "shape_to_string" =
  print_endline (shape_to_string [2; 3; 4]);
  [%expect {| 2x3x4x |}];
  print_endline (shape_to_string []);
  [%expect {| |}];
  print_endline (shape_to_string [1]);
  [%expect {| 1x |}]

type value_type = Tensor_type of shape * tensor_element_type

let value_type_to_string = function
  | Tensor_type (shape, element_type) ->
      "tensor<" ^ shape_to_string shape
      ^ tensor_element_type_to_string element_type
      ^ ">"

let%expect_test "value_type_to_string" =
  print_endline (value_type_to_string (Tensor_type ([2; 3], F32)));
  [%expect {| tensor<2x3xf32> |}];
  print_endline (value_type_to_string (Tensor_type ([], I64)));
  [%expect {| tensor<i64> |}];
  print_endline (value_type_to_string (Tensor_type ([1; 2; 3], U32)));
  [%expect {| tensor<1x2x3xui32> |}]

type annotated_value = string * value_type

type op =
  { inputs: annotated_value list
  ; outputs: annotated_value list
  ; name: string
  ; attributes: (string * string) list
  ; anonymous_functions: func list
  ; call: bool
  ; reduce_info: string option }

and func =
  { id: string
  ; inputs: annotated_value list
  ; outputs: value_type list
  ; body: op list }

let rec op_to_string (op : op) =
  let outputs =
    String.concat ", " (List.map (fun (output, _) -> "%" ^ output) op.outputs)
    ^ if List.length op.outputs = 0 then "" else " = "
  in
  let inputs =
    String.concat ", " (List.map (fun (input, _) -> "%" ^ input) op.inputs)
  in
  let anonymous_functions =
    List.map (fun func -> func_to_anonymous_string func) op.anonymous_functions
  in
  let anonymous_functions =
    if List.is_empty anonymous_functions then ""
    else "(" ^ String.concat ", " anonymous_functions ^ ")"
  in
  let attributes =
    if List.is_empty op.attributes then ""
    else
      " {\n"
      ^ String.concat ",\n"
          (List.map (fun (attr, value) -> attr ^ " = " ^ value) op.attributes)
      ^ "\n}"
  in
  let input_types =
    String.concat ", "
      (List.map (fun (_, t) -> value_type_to_string t) op.inputs)
  in
  let output_types =
    "("
    ^ String.concat ", "
        (List.map (fun (_, t) -> value_type_to_string t) op.outputs)
    ^ ")"
  in
  let signature = "(" ^ input_types ^ ") -> " ^ output_types ^ "" in
  let call = if op.call then "call @" ^ op.name else "\"" ^ op.name ^ "\"" in
  match op.reduce_info with
  | Some op_name ->
      let data_input, _ = List.hd op.inputs in
      let init_input, _ = List.nth op.inputs 1 in
      let dims = List.assoc "dimensions" op.attributes in
      outputs
      ^ "stablehlo.reduce(%"
      ^ data_input
      ^ " init: %"
      ^ init_input
      ^ ") applies "
      ^ op_name
      ^ " across dimensions = " ^ dims ^ " : " ^ signature
  | None ->
      outputs ^ call ^ "(" ^ inputs ^ ")" ^ anonymous_functions ^ attributes ^ " : " ^ signature

and func_to_anonymous_string func =
  let inputs =
    String.concat ", "
      (List.map
         (fun (name, t) -> "%" ^ name ^ " : " ^ value_type_to_string t)
         func.inputs )
  in
  let body = String.concat "\n" (List.map op_to_string func.body) in
  "{^a(" ^ inputs ^ "):\n" ^ body ^ "\n}"

let%expect_test "op_to_string" =
  let op = {
    inputs = [("input", Tensor_type ([2; 3], F32))];
    outputs = [("output", Tensor_type ([2; 3], F32))];
    name = "stablehlo.add";
    attributes = [];
    anonymous_functions = [];
    call = false;
    reduce_info = None
  } in
  print_endline (op_to_string op);
  [%expect {| %output = "stablehlo.add"(%input) : (tensor<2x3xf32>) -> (tensor<2x3xf32>) |}];

  let op_with_attrs = {
    inputs = [("input", Tensor_type ([2; 3], F32))];
    outputs = [("output", Tensor_type ([2; 3], F32))];
    name = "stablehlo.constant";
    attributes = [("value", "dense<1.0> : tensor<f32>")];
    anonymous_functions = [];
    call = false;
    reduce_info = None
  } in
  print_endline (op_to_string op_with_attrs);
  [%expect {|
    %output = "stablehlo.constant"(%input) {
    value = dense<1.0> : tensor<f32>
    } : (tensor<2x3xf32>) -> (tensor<2x3xf32>)
    |}];

  let op_with_reduce = {
    inputs = [
      ("data", Tensor_type ([2; 3], F32));
      ("init", Tensor_type ([], F32))
    ];
    outputs = [("output", Tensor_type ([3], F32))];
    name = "stablehlo.reduce";
    attributes = [("dimensions", "[0]")];
    anonymous_functions = [];
    call = false;
    reduce_info = Some "stablehlo.add"
  } in
  print_endline (op_to_string op_with_reduce);
  [%expect {| %output = stablehlo.reduce(%data init: %init) applies stablehlo.add across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> (tensor<3xf32>) |}]

let%expect_test "func_to_anonymous_string" =
  let func = {
    id = "add";
    inputs = [("x", Tensor_type ([2; 3], F32)); ("y", Tensor_type ([2; 3], F32))];
    outputs = [Tensor_type ([2; 3], F32)];
    body = [{
      inputs = [("x", Tensor_type ([2; 3], F32)); ("y", Tensor_type ([2; 3], F32))];
      outputs = [("output", Tensor_type ([2; 3], F32))];
      name = "stablehlo.add";
      attributes = [];
      anonymous_functions = [];
      call = false;
      reduce_info = None
    }]
  } in
  print_endline (func_to_anonymous_string func);
  [%expect {|
    {^a(%x : tensor<2x3xf32>, %y : tensor<2x3xf32>):
    %output = "stablehlo.add"(%x, %y) : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>)
    }
    |}]

let func_to_string (func : func) =
  let inputs =
    String.concat ", "
      (List.map
         (fun (name, t) -> "%" ^ name ^ " : " ^ value_type_to_string t)
         func.inputs )
  in
  let outputs =
    "(" ^ String.concat ", " (List.map value_type_to_string func.outputs) ^ ")"
  in
  let body = String.concat "\n" (List.map op_to_string func.body) in
  "func.func @" ^ func.id ^ "(" ^ inputs ^ ") -> " ^ outputs ^ " {\n" ^ body
  ^ "\n}"

let%expect_test "func_to_string" =
  let func = {
    id = "main";
    inputs = [("x", Tensor_type ([2; 3], F32)); ("y", Tensor_type ([2; 3], F32))];
    outputs = [Tensor_type ([2; 3], F32)];
    body = [{
      inputs = [("x", Tensor_type ([2; 3], F32)); ("y", Tensor_type ([2; 3], F32))];
      outputs = [("output", Tensor_type ([2; 3], F32))];
      name = "stablehlo.add";
      attributes = [];
      anonymous_functions = [];
      call = false;
      reduce_info = None
    }]
  } in
  print_endline (func_to_string func);
  [%expect {|
    func.func @main(%x : tensor<2x3xf32>, %y : tensor<2x3xf32>) -> (tensor<2x3xf32>) {
    %output = "stablehlo.add"(%x, %y) : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>)
    }
    |}]
