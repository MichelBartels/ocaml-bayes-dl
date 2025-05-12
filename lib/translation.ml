module Tensor = Device_api.Tensor

let attribute_of_comparison_direction direction =
  ( "comparison_direction"
  , "#stablehlo<comparison_direction "
    ^ ( match direction with
      | Var.Eq ->
          "EQ"
      | Ne ->
          "NE"
      | Ge ->
          "GE"
      | Gt ->
          "GT"
      | Le ->
          "LE"
      | Lt ->
          "LT" )
    ^ ">" )

let%expect_test "attribute_of_comparison_direction" =
  let test_direction dir =
    let attr = attribute_of_comparison_direction dir in
    print_endline (fst attr ^ " = " ^ snd attr)
  in
  test_direction Var.Eq;
  test_direction Var.Ne;
  test_direction Var.Ge;
  test_direction Var.Gt;
  test_direction Var.Le;
  test_direction Var.Lt;
  [%expect {|
    comparison_direction = #stablehlo<comparison_direction EQ>
    comparison_direction = #stablehlo<comparison_direction NE>
    comparison_direction = #stablehlo<comparison_direction GE>
    comparison_direction = #stablehlo<comparison_direction GT>
    comparison_direction = #stablehlo<comparison_direction LE>
    comparison_direction = #stablehlo<comparison_direction LT>
  |}]


module Func : sig
  type ('a, 'b) t =
    { inputs: 'a Value_type.t
    ; parameter_names: string list
    ; outputs: 'b Var.t
    ; name: string }
end = struct
  type ('a, 'b) t =
    { inputs: 'a Value_type.t
    ; parameter_names: string list
    ; outputs: 'b Var.t
    ; name: string }
end

module AnnotatedValueSet = Set.Make (struct
  type t = string * Stable_hlo.value_type

  let compare = Stdlib.compare
end)

module VarMap = struct
  type 'a t = (Var.any * 'a) list

  let empty = []

  let add var value map =
    if List.mem_assoc var map then map else (var, value) :: map

  let mem = List.mem_assoc

  let find = List.assoc

  let bindings map = map
end

let vars_to_ops vars =
  let rec aux : type a b.
         Stable_hlo.annotated_value list
         * (Stable_hlo.op option * Stable_hlo.annotated_value) VarMap.t
      -> (a, b) Var.u
      -> Stable_hlo.annotated_value list
         * (Stable_hlo.op option * Stable_hlo.annotated_value) VarMap.t =
   fun (prev_outputs, cache) var ->
    if VarMap.mem (Var.Any var) cache then
      ((snd @@ VarMap.find (Var.Any var) cache) :: prev_outputs, cache)
    else
      let add var = VarMap.add (Var.Any var) in
      match var with
      | Add (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.add"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Subtract (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.subtract"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Multiply (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.multiply"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Divide (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.divide"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Negate var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.negate"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Abs var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.abs"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Ln var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.log"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Ln_1_plus var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.log_plus_one"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Exponential var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.exponential"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Pow (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.power"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Argument _ ->
          (Var.to_annotated_value var :: prev_outputs, cache)
      | Compare (lhs, direction, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.compare"
              ; attributes= [attribute_of_comparison_direction direction]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Min (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.minimum"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Max (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.maximum"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Constant tensor ->
          let output = Var.to_annotated_value var in
          let repr = Tensor.to_string tensor in
          let signature =
            Stable_hlo.value_type_to_string @@ Value_type.tensor_to_stable_hlo
            @@ Value_type.of_tensor tensor
          in
          let repr = Printf.sprintf "dense<%s> : %s" repr signature in
          let op =
            Stable_hlo.
              { inputs= []
              ; outputs= [output]
              ; name= "stablehlo.constant"
              ; attributes= [("value", repr)]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | BroadcastScalarConstant (value_type, scalar) ->
          let output = Var.to_annotated_value var in
          let repr = Tensor.value_to_string (snd value_type) scalar in
          let signature =
            Stable_hlo.value_type_to_string @@ Value_type.tensor_to_stable_hlo
            @@ value_type
          in
          let repr = Printf.sprintf "dense<%s> : %s" repr signature in
          let op =
            Stable_hlo.
              { inputs= []
              ; outputs= [output]
              ; name= "stablehlo.constant"
              ; attributes= [("value", repr)]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | DotProduct
          ( lhs
          , rhs
          , lhs_contracting_dims
          , rhs_contracting_dims
          , lhs_batching_dims
          , rhs_batching_dims ) ->
          let output = Var.to_annotated_value var in
          let dims_to_string dims =
            "[" ^ String.concat "," (List.map string_of_int dims) ^ "]"
          in
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.dot_general"
              ; attributes=
                  [ ( "dot_dimension_numbers"
                    , "#stablehlo.dot<\nlhs_batching_dimensions = "
                      ^ dims_to_string lhs_batching_dims
                      ^ ",\nrhs_batching_dimensions = "
                      ^ dims_to_string rhs_batching_dims
                      ^ ",\nlhs_contracting_dimensions = "
                      ^ dims_to_string lhs_contracting_dims
                      ^ ",\nrhs_contracting_dimensions = "
                      ^ dims_to_string rhs_contracting_dims
                      ^ "\n>" ) ]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Random (_, a, b, shape, distribution) ->
          let a, cache = aux ([], cache) a in
          let b, cache = aux ([], cache) b in
          let shape, cache = aux ([], cache) shape in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= a @ b @ shape
              ; outputs= [output]
              ; name= "stablehlo.rng"
              ; attributes=
                  [ ( "rng_distribution"
                    , "#stablehlo<rng_distribution "
                      ^ ( match distribution with
                        | Var.Uniform ->
                            "UNIFORM"
                        | Normal ->
                            "NORMAL" )
                      ^ ">" ) ]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | DiffVar (_, var) ->
          aux (prev_outputs, cache) var
      | BroadcastInDim (var', new_dims) ->
          let var'', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var''
              ; outputs= [output]
              ; name= "stablehlo.broadcast_in_dim"
              ; attributes=
                  [ ( "broadcast_dimensions"
                    , let shape = Var.shape var' in
                      if List.is_empty shape then "array<i64>"
                      else
                        "array<i64: "
                        ^ String.concat ","
                            (List.init
                               (List.length (Var.shape var'))
                               (fun i ->
                                 string_of_int (i + List.length new_dims) ) )
                        ^ ">" ) ]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Transpose (var', permutation) ->
          if List.length permutation = 0 then
            Stdlib.(
              if List.length (Var.shape var') = 0 then
                aux (prev_outputs, cache) var'
              else failwith "expected non-empty permutation" )
          else
            let var'', cache = aux ([], cache) var' in
            let output = Var.to_annotated_value var in
            let op =
              Stable_hlo.
                { inputs= var''
                ; outputs= [output]
                ; name= "stablehlo.transpose"
                ; attributes=
                    [ ( "permutation"
                      , "array<i64: "
                        ^ String.concat "," (List.map string_of_int permutation)
                        ^ ">" ) ]
                ; anonymous_functions= []
                ; call= false
                ; reduce_info= None }
            in
            (output :: prev_outputs, add var (Some op, output) cache)
      | Tanh var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.tanh"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Sum (var', dimensions) ->
          let var', cache = aux ([], cache) var' in
          let initial, cache =
            aux ([], cache) (Var.BroadcastScalarConstant (([], F32), 0.0))
          in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var' @ initial
              ; outputs= [output]
              ; name= "stablehlo.reduce"
              ; attributes= [("dimensions", "[" ^ String.concat ", " (List.map string_of_int dimensions) ^ "]")]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= Some "stablehlo.add" }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | RightShift (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.shift_right_logical"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | LeftShift (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.shift_left"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Bitcast (var', new_type) ->
          let var', cache = aux ([], cache) var' in
          let output_id, Tensor_type (shape, _) = Var.to_annotated_value var in
          let output =
            (output_id, Value_type.tensor_to_stable_hlo (shape, new_type))
          in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.bitcast_convert"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Convert (var', new_type) ->
          let var', cache = aux ([], cache) var' in
          let output_id, Tensor_type (shape, _) = Var.to_annotated_value var in
          let output =
            (output_id, Value_type.tensor_to_stable_hlo (shape, new_type))
          in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.convert"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | NoGrad var' ->
          aux (prev_outputs, cache) var'
      | Or (lhs, rhs) ->
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.or"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Iota (index, _) ->
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= []
              ; outputs= [output]
              ; name= "stablehlo.iota"
              ; attributes= [("iota_dimension", string_of_int index ^ " : i64")]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Reshape (var', _) ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.reshape"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Sin var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.sine"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Cos var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.cosine"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Concatenate (vars, axis) ->
          let cache, vars =
            List.fold_left_map
              (fun cache var ->
                let vars, cache = aux ([], cache) var in
                (cache, List.hd vars) )
              cache vars
          in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= vars
              ; outputs= [output]
              ; name= "stablehlo.concatenate"
              ; attributes= [("dimension", string_of_int axis)]
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Select (condition, lhs, rhs) ->
          let condition, cache = aux ([], cache) condition in
          let lhs, cache = aux ([], cache) lhs in
          let rhs, cache = aux ([], cache) rhs in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= condition @ lhs @ rhs
              ; outputs= [output]
              ; name= "stablehlo.select"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Sqrt var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.sqrt"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | OptimizationBarrier var' ->
          let var', cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var'
              ; outputs= [output]
              ; name= "stablehlo.optimization_barrier"
              ; attributes= []
              ; anonymous_functions= []
              ; call= false
              ; reduce_info= None }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
  in
  let outputs, cache = Var.List.fold_left {f= aux} ([], VarMap.empty) vars in
  (outputs, VarMap.bindings cache |> List.map snd |> List.map fst |> List.rev)

let annotated_values_to_return_op values =
  Stable_hlo.
    { inputs= values
    ; outputs= []
    ; name= "func.return"
    ; attributes= []
    ; anonymous_functions= []
    ; call= false
    ; reduce_info= None }

let create_func : type a b.
    a Value_type.t -> (a Var.t -> b Var.t) -> (a, b) Func.t =
 fun inputs body ->
  let open Hlist.Map (Value_type.List) (Var.List) in
  let args = Var.arg_of_value_type inputs in
  let outputs = body args in
  let parameter_names = Var.get_args args in
  let parameter_names = List.map string_of_int parameter_names in
  {inputs; parameter_names; outputs; name= "main"}

let func_to_stable_hlo (func : ('a, 'b) Func.t) =
  let outputs, ops = vars_to_ops func.outputs in
  let ops = List.filter_map (fun x -> x) ops in
  let inputs = Value_type.to_stable_hlo func.inputs |> List.rev in
  let inputs = List.combine func.parameter_names inputs in
  let outputs = List.rev outputs in
  let return_ops = annotated_values_to_return_op outputs in
  let outputs = List.map snd outputs in
  Stable_hlo.{id= func.name; inputs; outputs; body= ops @ [return_ops]}

module StringMap = Map.Make (String)

let translate entry =
  Stable_hlo.func_to_string @@ func_to_stable_hlo entry

let%expect_test "create_func" =
  let inputs = Value_type.List.E ([2; 2], F32) in
  let func = create_func inputs (fun x -> Var.List.E (Var.Add (Var.List.unwrap x, Var.List.unwrap x))) in
  print_endline ("Name: " ^ func.name);
  print_endline ("Parameter names: " ^ String.concat ", " func.parameter_names);
  print_endline (Var.to_string (Var.List.unwrap func.outputs));
  [%expect {|
    Name: main
    Parameter names: 0
    (arg0 + arg0)
    |}]

let%expect_test "func_to_stable_hlo" =
  let inputs = Value_type.List.E ([2; 2], F32) in
  let func = create_func inputs (fun x -> Var.List.E (Var.Add (Var.List.unwrap x, Var.List.unwrap x))) in
  let hlo = func_to_stable_hlo func in
  print_endline ("ID: " ^ hlo.id);
  print_endline ("Inputs: " ^ String.concat ", " (List.map (fun (name, _) -> name) hlo.inputs));
  print_endline ("Outputs: " ^ String.concat ", " (List.map Stable_hlo.value_type_to_string hlo.outputs));
  [%expect {|
    ID: main
    Inputs: 1
    Outputs: tensor<2x2xf32>
    |}]

let%expect_test "translate" =
  let inputs = Value_type.List.E ([2; 2], F32) in
  let func = create_func inputs (fun x -> Var.List.E (Var.Add (Var.List.unwrap x, Var.List.unwrap x))) in
  let result = translate func in
  print_endline result;
  [%expect {|
    func.func @main(%3 : tensor<2x2xf32>) -> (tensor<2x2xf32>) {
    %4 = "stablehlo.add"(%3, %3) : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    "func.return"(%4) : (tensor<2x2xf32>) -> ()
    }
    |}]
