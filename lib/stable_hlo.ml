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

type value_type = Tensor_type of shape * tensor_element_type

let value_type_to_string = function
  | Tensor_type (shape, element_type) ->
      "tensor<" ^ shape_to_string shape
      ^ tensor_element_type_to_string element_type
      ^ ">"

type annotated_value = string * value_type

type op =
  { inputs: annotated_value list
  ; outputs: annotated_value list
  ; name: string
  ; attributes: (string * string) list
  ; anonymous_functions: func list
  ; call: bool }

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
  outputs ^ call ^ "(" ^ inputs ^ ")" ^ anonymous_functions ^ attributes ^ " : "
  ^ signature

and func_to_anonymous_string func =
  let inputs =
    String.concat ", "
      (List.map
         (fun (name, t) -> "%" ^ name ^ " : " ^ value_type_to_string t)
         func.inputs )
  in
  let body = String.concat "\n" (List.map op_to_string func.body) in
  "{^a(" ^ inputs ^ "):\n" ^ body ^ "\n}"

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
