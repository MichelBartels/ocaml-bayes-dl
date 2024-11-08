type id = int

let current_id = ref 0

let new_id () =
  let id = !current_id in
  current_id := id + 1 ;
  id

let tag x = (string_of_int @@ new_id (), x)

type f32 = F32

type i1 = I1

type i64 = I64

type _ tensor = F32 : f32 tensor | I1 : i1 tensor | I64 : i64 tensor

type shape = int list

let tensor_element_type_to_stable_hlo :
    type a. a tensor -> Stable_hlo.tensor_element_type = function
  | F32 ->
      Stable_hlo.F32
  | I1 ->
      Stable_hlo.I1
  | I64 ->
      Stable_hlo.I64

type 'a tagged = id * 'a

type comparison_direction = Eq | Ne | Ge | Gt | Le | Lt

let attribute_of_comparison_direction direction =
  ( "comparison_direction"
  , "#stablehlo<comparison_direction "
    ^ ( match direction with
      | Eq ->
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

type distribution = Uniform | Normal

module rec Var : sig
  type _ t =
    | Add : 'a tensor t * 'a tensor t -> 'a tensor t
    | Subtract : 'a tensor t * 'a tensor t -> 'a tensor t
    | Multiply : 'a tensor t * 'a tensor t -> 'a tensor t
    | Abs : 'a tensor t -> 'a tensor t
    | Argument : id * 'a tensor ValueType.t -> 'a tensor t
    | Compare : 'a tensor t * comparison_direction * 'a tensor t -> i1 tensor t
    | Constant : 'a tensor ValueType.t * string -> 'a tensor t
    | Random :
        'a tensor ValueType.t
        * f32 tensor t
        * f32 tensor t
        * i64 tensor t
        * distribution
        -> 'a tensor t
    | [] : unit VarList.t t
    | ( :: ) : 'a t * 'b VarList.t t -> ('a t -> 'b) VarList.t t
    | DiffVar : id * 'a tensor t -> 'a tensor t
    | DiffConst : 'a tensor t -> 'a tensor t

  val to_var_list : 'a VarList.t t -> 'a VarList.t

  val from_var_list : 'a VarList.t -> 'a VarList.t t

  val to_annotated_values : 'a t -> (string * Stable_hlo.value_type) list

  val to_annotated_value : 'a tensor t -> string * Stable_hlo.value_type

  val length : 'a Var.t -> int
end = struct
  type _ t =
    | Add : 'a tensor t * 'a tensor t -> 'a tensor t
    | Subtract : 'a tensor t * 'a tensor t -> 'a tensor t
    | Multiply : 'a tensor t * 'a tensor t -> 'a tensor t
    | Abs : 'a tensor t -> 'a tensor t
    | Argument : id * 'a tensor ValueType.t -> 'a tensor t
    | Compare : 'a tensor t * comparison_direction * 'a tensor t -> i1 tensor t
    | Constant : 'a tensor ValueType.t * string -> 'a tensor t
    | Random :
        'a tensor ValueType.t
        * f32 tensor t
        * f32 tensor t
        * i64 tensor t
        * distribution
        -> 'a tensor t
    | [] : unit VarList.t t
    | ( :: ) : 'a t * 'b VarList.t t -> ('a t -> 'b) VarList.t t
    | DiffVar : id * 'a tensor t -> 'a tensor t
    | DiffConst : 'a tensor t -> 'a tensor t

  let rec to_var_list : type a. a VarList.t t -> a VarList.t = function
    | [] ->
        []
    | hd :: tl ->
        hd :: to_var_list tl

  let rec from_var_list : type a. a VarList.t -> a VarList.t t = function
    | [] ->
        []
    | hd :: tl ->
        hd :: from_var_list tl

  let to_annotated_values var =
    List.map tag (ValueType.to_stable_hlo @@ ValueType.of_var var)

  let to_annotated_value var =
    match var with
    | Argument (id, value_type) ->
        (string_of_int id, ValueType.tensor_to_stable_hlo value_type)
    | _ ->
        tag @@ ValueType.tensor_to_stable_hlo @@ ValueType.of_var var

  let rec length : type a. a Var.t -> int = function
    | [] ->
        0
    | x :: xs ->
        length x + length xs
    | _ ->
        1
end

and ValueType : sig
  type _ t =
    | Tensor_type : shape * 'a tensor -> 'a tensor t
    | List_type : 'a ValueTypeList.t -> 'a VarList.t t

  val tensor_to_stable_hlo : 'a tensor t -> Stable_hlo.value_type

  val to_stable_hlo : 'a t -> Stable_hlo.value_type list

  val of_var : 'a Var.t -> 'a t

  val to_arg : 'a t -> 'a Var.t
end = struct
  type _ t =
    | Tensor_type : shape * 'a tensor -> 'a tensor t
    | List_type : 'a ValueTypeList.t -> 'a VarList.t t

  let tensor_to_stable_hlo : type a. a tensor t -> Stable_hlo.value_type =
    function
    | Tensor_type (shape, tensor_element_type) ->
        Stable_hlo.Tensor_type
          (shape, tensor_element_type_to_stable_hlo tensor_element_type)

  let rec to_stable_hlo : type a. a t -> Stable_hlo.value_type list = function
    | Tensor_type (shape, tensor_element_type) ->
        [tensor_to_stable_hlo (Tensor_type (shape, tensor_element_type))]
    | List_type l ->
        ValueTypeList.map_to_list {f= to_stable_hlo} l |> List.concat

  let rec of_var : type a. a Var.t -> a t = function
    | Add (lhs, _) ->
        of_var lhs
    | Subtract (lhs, _) ->
        of_var lhs
    | Multiply (lhs, _) ->
        of_var lhs
    | Abs var ->
        of_var var
    | Argument (_, value_type) ->
        value_type
    | Compare (a, _, _) ->
        let (Tensor_type (shape, _)) = of_var a in
        Tensor_type (shape, I1)
    | Constant (value_type, _) ->
        value_type
    | Random (value_type, _, _, _, _) ->
        value_type
    | [] ->
        List_type []
    | x :: xs ->
        let l = Var.to_var_list (x :: xs) in
        let open Hlist.Map (VarList) (ValueTypeList) in
        let l = map {f= of_var} l in
        List_type l
    | DiffVar (_, v) ->
        of_var v
    | DiffConst v ->
        of_var v

  let rec to_arg : type a. a t -> a Var.t = function
    | Tensor_type _ as t ->
        Var.Argument (new_id (), t)
    | List_type l ->
        let open Hlist.Map (ValueTypeList) (VarList) in
        let l = map {f= to_arg} l in
        Var.from_var_list l
end

and VarList : (Hlist.S with type 'a u = 'a Var.t and type 'a v = 'a Var.t) =
Hlist.Make (struct
  type 'a t = 'a Var.t

  type 'a tag = 'a Var.t
end)

and ValueTypeList :
  (Hlist.S with type 'a u = 'a ValueType.t and type 'a v = 'a Var.t) =
Hlist.Make (struct
  type 'a t = 'a ValueType.t

  type 'a tag = 'a Var.t
end)

and Func : sig
  type ('a, 'b) t =
    { inputs: 'a ValueType.t
    ; parameter_names: string list
    ; outputs: 'b Var.t
    ; name: string }
end = struct
  type ('a, 'b) t =
    { inputs: 'a ValueType.t
    ; parameter_names: string list
    ; outputs: 'b Var.t
    ; name: string }
end

let shape_of_var var =
  ValueType.of_var var |> function Tensor_type (shape, _) -> shape

module AnnotatedValueSet = Set.Make (struct
  type t = string * Stable_hlo.value_type

  let compare = Stdlib.compare
end)

type any_var = Any_var : 'a Var.t -> any_var

module VarMap = struct
  type 'a t = (any_var * 'a) list

  let empty = []

  let add var value map = (var, value) :: map

  let mem var map = List.assoc_opt var map |> Option.is_some

  let find var map = List.assoc var map

  let bindings map = map
end

let vars_to_ops vars =
  let rec aux :
      type a.
         Stable_hlo.annotated_value list
         * (Stable_hlo.op option * Stable_hlo.annotated_value) VarMap.t
      -> a Var.t
      -> Stable_hlo.annotated_value list
         * (Stable_hlo.op option * Stable_hlo.annotated_value) VarMap.t =
   fun (prev_outputs, cache) var ->
    if VarMap.mem (Any_var var) cache then
      ((snd @@ VarMap.find (Any_var var) cache) :: prev_outputs, cache)
    else
      let add var = VarMap.add (Any_var var) in
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Abs var' ->
          let var, cache = aux ([], cache) var' in
          let output = Var.to_annotated_value var' in
          let op =
            Stable_hlo.
              { inputs= var
              ; outputs= [output]
              ; name= "stablehlo.abs"
              ; attributes= []
              ; call= false }
          in
          (output :: prev_outputs, add var' (Some op, output) cache)
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
              ; call= false }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Constant (_, repr) ->
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= []
              ; outputs= [output]
              ; name= "stablehlo.constant"
              ; attributes= [("value", repr)]
              ; call= false }
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
                        | Uniform ->
                            "UNIFORM"
                        | Normal ->
                            "NORMAL" )
                      ^ ">" ) ]
              ; call= false }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | [] ->
          (prev_outputs, cache)
      | hd :: tl ->
          let hd, cache = aux ([], cache) hd in
          aux (hd @ prev_outputs, cache) tl
      | DiffVar (_, var) ->
          aux (prev_outputs, cache) var
      | DiffConst var ->
          aux (prev_outputs, cache) var
  in
  let outputs, cache = aux ([], VarMap.empty) vars in
  (outputs, VarMap.bindings cache |> List.map snd |> List.map fst |> List.rev)

let annotated_values_to_return_op values =
  Stable_hlo.
    { inputs= values
    ; outputs= []
    ; name= "func.return"
    ; attributes= []
    ; call= false }

let rec get_args : type a. a Var.t -> id list = function
  | Add (a, b) ->
      get_args a @ get_args b
  | Subtract (a, b) ->
      get_args a @ get_args b
  | Multiply (a, b) ->
      get_args a @ get_args b
  | Abs a ->
      get_args a
  | Argument (id, _) ->
      [id]
  | Compare (a, _, b) ->
      get_args a @ get_args b
  | Constant _ ->
      []
  | Random (_, a, b, c, _) ->
      get_args a @ get_args b @ get_args c
  | [] ->
      []
  | x :: xs ->
      let l = Var.to_var_list (x :: xs) in
      VarList.fold_left {f= (fun args var -> get_args var @ args)} [] l
  | DiffVar (_, var) ->
      get_args var
  | DiffConst var ->
      get_args var

let create_func :
    type a b. a ValueType.t -> (a Var.t -> b Var.t) -> (a, b) Func.t =
 fun inputs body ->
  let open Hlist.Map (ValueTypeList) (VarList) in
  let args = ValueType.to_arg inputs in
  let outputs = body args in
  let parameter_names = get_args args in
  let parameter_names = List.sort Stdlib.compare parameter_names |> List.rev in
  let parameter_names = List.map string_of_int parameter_names in
  {inputs; parameter_names; outputs; name= "fn" ^ string_of_int (new_id ())}

let func_to_stable_hlo (func : ('a, 'b) Func.t) =
  let outputs, ops = vars_to_ops func.outputs in
  let ops = List.filter_map (fun x -> x) ops in
  let inputs = ValueType.to_stable_hlo func.inputs in
  let inputs = List.combine func.parameter_names inputs in
  let return_ops = annotated_values_to_return_op outputs in
  let outputs = List.map snd outputs in
  Stable_hlo.{id= func.name; inputs; outputs; body= ops @ [return_ops]}

module StringMap = Map.Make (String)

let compile entry =
  let rec all_funcs :
      type a. string StringMap.t -> a Var.t -> string StringMap.t =
   fun cache var ->
    match var with
    | Var.Add (a, b) ->
        all_funcs (all_funcs cache a) b
    | Subtract (a, b) ->
        all_funcs (all_funcs cache a) b
    | Multiply (a, b) ->
        all_funcs (all_funcs cache a) b
    | Abs a ->
        all_funcs cache a
    | Argument _ ->
        cache
    | Compare (a, _, b) ->
        all_funcs (all_funcs cache a) b
    | Constant _ ->
        cache
    | Random (_, a, b, c, _) ->
        all_funcs (all_funcs (all_funcs cache a) b) c
    | [] ->
        cache
    | x :: xs ->
        let l = Var.to_var_list (x :: xs) in
        VarList.fold_left {f= all_funcs} cache l
    | DiffVar (_, var) ->
        all_funcs cache var
    | DiffConst var ->
        all_funcs cache var
  in
  let main = func_to_stable_hlo entry |> Stable_hlo.func_to_string in
  let cache = StringMap.add entry.Func.name main StringMap.empty in
  let funcs = all_funcs cache entry.Func.outputs in
  StringMap.bindings funcs |> List.map snd |> String.concat "\n"
