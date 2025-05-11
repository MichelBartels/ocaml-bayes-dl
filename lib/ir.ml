type id = int

let current_id = ref 0

let new_id () =
  let id = !current_id in
  current_id := id + 1 ;
  id

let tag x = (string_of_int @@ new_id (), x)

type shape = int list

module Tensor = Device_api.Tensor

let tensor_element_type_to_stable_hlo : type a b.
    (a, b) Tensor.kind -> Stable_hlo.tensor_element_type = function
  | F32 ->
      Stable_hlo.F32
  | I1 ->
      Stable_hlo.I1
  | I64 ->
      Stable_hlo.I64
  | U32 ->
      Stable_hlo.U32
  | U64 ->
      Stable_hlo.U64
  | F64 ->
      Stable_hlo.F64

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
  type (_, _) u =
    | Add : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Subtract : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Multiply : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Divide : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Negate : ('a, 'b) u -> ('a, 'b) u
    | Abs : ('a, 'b) u -> ('a, 'b) u
    | Ln : ('a, 'b) u -> ('a, 'b) u
    | Ln_1_plus : ('a, 'b) u -> ('a, 'b) u
    | Exponential : ('a, 'b) u -> ('a, 'b) u
    | Pow : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Argument : id * ('a, 'b) ValueType.u -> ('a, 'b) u
    | Compare :
        ('a, 'b) u * comparison_direction * ('a, 'b) u
        -> (Tensor.i1, bool) u
    | Min : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Max : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Constant : ('a, 'b) Tensor.t -> ('a, 'b) u
    | BroadcastScalarConstant : ('a, 'b) ValueType.u * 'b -> ('a, 'b) u
    | DotProduct :
        ('a, 'b) u * ('a, 'b) u * int list * int list * int list * int list
        -> ('a, 'b) u
    | Random :
        ('a, 'b) ValueType.u
        * (Tensor.f32, float) u
        * (Tensor.f32, float) u
        * (Tensor.i64, Signed.Int64.t) u
        * distribution
        -> ('a, 'b) u
    | DiffVar : id * ('a, 'b) u -> ('a, 'b) u
    | BroadcastInDim : ('a, 'b) u * int list -> ('a, 'b) u
    | Transpose : ('a, 'b) u * int list -> ('a, 'b) u
    | Tanh : ('a, 'b) u -> ('a, 'b) u
    | Sum : (Tensor.f32, float) u * int list -> (Tensor.f32, float) u
    | RightShift :
        (Tensor.u64, Unsigned.uint64) u * (Tensor.u64, Unsigned.uint64) u
        -> (Tensor.u64, Unsigned.uint64) u
    | LeftShift :
        (Tensor.u64, Unsigned.uint64) u * (Tensor.u64, Unsigned.uint64) u
        -> (Tensor.u64, Unsigned.uint64) u
    | Bitcast : ('a, 'b) u * ('c, 'd) Tensor.kind -> ('c, 'd) u
    | Convert : ('a, 'b) u * ('c, 'd) Tensor.kind -> ('c, 'd) u
    | NoGrad : ('a, 'b) u -> ('a, 'b) u
    | Or :
        (Tensor.u64, Unsigned.uint64) u * (Tensor.u64, Unsigned.uint64) u
        -> (Tensor.u64, Unsigned.uint64) u
    | Iota : int * int list -> (Tensor.u64, Unsigned.uint64) u
    | Reshape : ('a, 'b) u * int list -> ('a, 'b) u
    | Sin : ('a, 'b) u -> ('a, 'b) u
    | Cos : ('a, 'b) u -> ('a, 'b) u
    | Concatenate : ('a, 'b) u list * int -> ('a, 'b) u
    | Select : (Tensor.i1, bool) u * ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Sqrt : ('a, 'b) u -> ('a, 'b) u
    | OptimizationBarrier : ('a, 'b) u -> ('a, 'b) u

  type any = Any : ('a, 'b) u -> any

  module List : Hlist.S with type ('a, 'b) u = ('a, 'b) u

  type 'a t = 'a List.t

  val to_annotated_values : 'a t -> (string * Stable_hlo.value_type) list

  val to_annotated_value : ('a, 'b) u -> string * Stable_hlo.value_type

  val length : 'a Var.t -> int

  val get_args : 'a Var.t -> id list

  type map2_fn = {f: 'a 'b. ('a, 'b) u -> ('a, 'b) u -> ('a, 'b) u}

  val map2 : map2_fn -> 'a t -> 'a t -> 'a t

  type 'a map2_acc_fn = 'a List.map2_acc_fn

  type 'a map_acc_fn = {f: 'b 'c. ('b, 'c) u -> 'a -> ('b, 'c) u * 'a}

  val map_acc : 'b map_acc_fn -> 'a t -> 'b -> 'a t * 'b

  type map_fn = {f: 'a 'b. ('a, 'b) u -> ('a, 'b) u}

  val map : map_fn -> 'a t -> 'a t

  val to_string : ('a, 'b) u -> string
end = struct
  type (_, _) u =
    | Add : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Subtract : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Multiply : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Divide : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Negate : ('a, 'b) u -> ('a, 'b) u
    | Abs : ('a, 'b) u -> ('a, 'b) u
    | Ln : ('a, 'b) u -> ('a, 'b) u
    | Ln_1_plus : ('a, 'b) u -> ('a, 'b) u
    | Exponential : ('a, 'b) u -> ('a, 'b) u
    | Pow : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Argument : id * ('a, 'b) ValueType.u -> ('a, 'b) u
    | Compare :
        ('a, 'b) u * comparison_direction * ('a, 'b) u
        -> (Tensor.i1, bool) u
    | Min : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Max : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Constant : ('a, 'b) Tensor.t -> ('a, 'b) u
    | BroadcastScalarConstant : ('a, 'b) ValueType.u * 'b -> ('a, 'b) u
    | DotProduct :
        ('a, 'b) u * ('a, 'b) u * int list * int list * int list * int list
        -> ('a, 'b) u
    | Random :
        ('a, 'b) ValueType.u
        * (Tensor.f32, float) u
        * (Tensor.f32, float) u
        * (Tensor.i64, Signed.Int64.t) u
        * distribution
        -> ('a, 'b) u
    | DiffVar : id * ('a, 'b) u -> ('a, 'b) u
    | BroadcastInDim : ('a, 'b) u * int list -> ('a, 'b) u
    | Transpose : ('a, 'b) u * int list -> ('a, 'b) u
    | Tanh : ('a, 'b) u -> ('a, 'b) u
    | Sum : (Tensor.f32, float) u * int list -> (Tensor.f32, float) u
    | RightShift :
        (Tensor.u64, Unsigned.uint64) u * (Tensor.u64, Unsigned.uint64) u
        -> (Tensor.u64, Unsigned.uint64) u
    | LeftShift :
        (Tensor.u64, Unsigned.uint64) u * (Tensor.u64, Unsigned.uint64) u
        -> (Tensor.u64, Unsigned.uint64) u
    | Bitcast : ('a, 'b) u * ('c, 'd) Tensor.kind -> ('c, 'd) u
    | Convert : ('a, 'b) u * ('c, 'd) Tensor.kind -> ('c, 'd) u
    | NoGrad : ('a, 'b) u -> ('a, 'b) u
    | Or :
        (Tensor.u64, Unsigned.uint64) u * (Tensor.u64, Unsigned.uint64) u
        -> (Tensor.u64, Unsigned.uint64) u
    | Iota : int * int list -> (Tensor.u64, Unsigned.uint64) u
    | Reshape : ('a, 'b) u * int list -> ('a, 'b) u
    | Sin : ('a, 'b) u -> ('a, 'b) u
    | Cos : ('a, 'b) u -> ('a, 'b) u
    | Concatenate : ('a, 'b) u list * int -> ('a, 'b) u
    | Select : (Tensor.i1, bool) u * ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Sqrt : ('a, 'b) u -> ('a, 'b) u
    | OptimizationBarrier : ('a, 'b) u -> ('a, 'b) u

  module VarList : Hlist.S with type ('a, 'b) u = ('a, 'b) u =
  Hlist.Make (struct
    type ('a, 'b) t = ('a, 'b) u
  end)

  type any = Any : ('a, 'b) u -> any

  type 'a t = 'a VarList.t

  let to_annotated_values var =
    List.map tag (ValueType.to_stable_hlo @@ ValueType.of_vars var)

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

  let rec get_args : type a. a Var.t -> id list = function
    | E (Argument (id, _)) ->
        [id]
    | [] ->
        []
    | x :: xs ->
        let args = get_args x in
        let args' = get_args xs in
        args @ args'
    | _ ->
        failwith "expected nested list of arguments"

  type map2_fn = {f: 'a 'b. ('a, 'b) u -> ('a, 'b) u -> ('a, 'b) u}

  type 'a map2_acc_fn = 'a VarList.map2_acc_fn

  let map2_acc = VarList.map2_acc

  let map2 ({f} : map2_fn) a b =
    map2_acc {f= (fun a b () -> (f a b, ()))} a b () |> fst

  type 'a map_acc_fn = {f: 'b 'c. ('b, 'c) u -> 'a -> ('b, 'c) u * 'a}

  let map_acc ({f} : 'a map_acc_fn) a acc =
    map2_acc {f= (fun a _ acc -> f a acc)} a a acc

  type map_fn = {f: 'a 'b. ('a, 'b) u -> ('a, 'b) u}

  let map ({f} : map_fn) a = map2 {f= (fun a _ -> f a)} a a

  let rec to_string : type a b. (a, b) u -> string = function
    | Add (lhs, rhs) -> Printf.sprintf "(%s + %s)" (to_string lhs) (to_string rhs)
    | Subtract (lhs, rhs) -> Printf.sprintf "(%s - %s)" (to_string lhs) (to_string rhs)
    | Multiply (lhs, rhs) -> Printf.sprintf "(%s * %s)" (to_string lhs) (to_string rhs)
    | Divide (lhs, rhs) -> Printf.sprintf "(%s / %s)" (to_string lhs) (to_string rhs)
    | Negate var -> Printf.sprintf "(-%s)" (to_string var)
    | Abs var -> Printf.sprintf "|%s|" (to_string var)
    | Ln var -> Printf.sprintf "ln(%s)" (to_string var)
    | Ln_1_plus var -> Printf.sprintf "ln(1 + %s)" (to_string var)
    | Exponential var -> Printf.sprintf "exp(%s)" (to_string var)
    | Pow (lhs, rhs) -> Printf.sprintf "(%s ^ %s)" (to_string lhs) (to_string rhs)
    | Argument (id, _) -> Printf.sprintf "arg%d" id
    | Compare (lhs, direction, rhs) ->
        let op = match direction with
          | Eq -> "=="
          | Ne -> "!="
          | Ge -> ">="
          | Gt -> ">"
          | Le -> "<="
          | Lt -> "<"
        in
        Printf.sprintf "(%s %s %s)" (to_string lhs) op (to_string rhs)
    | Min (lhs, rhs) -> Printf.sprintf "min(%s, %s)" (to_string lhs) (to_string rhs)
    | Max (lhs, rhs) -> Printf.sprintf "max(%s, %s)" (to_string lhs) (to_string rhs)
    | Constant tensor -> Printf.sprintf "const(%s)" (Tensor.to_string tensor)
    | BroadcastScalarConstant (value_type, scalar) -> Printf.sprintf "const(%s)" (Tensor.value_to_string (snd value_type) scalar)
    | DotProduct (lhs, rhs, _, _, _, _) -> Printf.sprintf "dot(%s, %s)" (to_string lhs) (to_string rhs)
    | Random (_, a, b, shape, dist) ->
        let dist_str = match dist with Uniform -> "uniform" | Normal -> "normal" in
        Printf.sprintf "random(%s, %s, %s, %s)" (to_string a) (to_string b) (to_string shape) dist_str
    | DiffVar (id, var) -> Printf.sprintf "diff%d(%s)" id (to_string var)
    | BroadcastInDim (var, dims) -> Printf.sprintf "broadcast(%s, %s)" (to_string var) (String.concat "," (List.map string_of_int dims))
    | Transpose (var, perm) -> Printf.sprintf "transpose(%s, %s)" (to_string var) (String.concat "," (List.map string_of_int perm))
    | Tanh var -> Printf.sprintf "tanh(%s)" (to_string var)
    | Sum (var, dims) -> Printf.sprintf "sum(%s, %s)" (to_string var) (String.concat "," (List.map string_of_int dims))
    | RightShift (lhs, rhs) -> Printf.sprintf "(%s >> %s)" (to_string lhs) (to_string rhs)
    | LeftShift (lhs, rhs) -> Printf.sprintf "(%s << %s)" (to_string lhs) (to_string rhs)
    | Bitcast (var, _) -> Printf.sprintf "bitcast(%s)" (to_string var)
    | Convert (var, _) -> Printf.sprintf "convert(%s)" (to_string var)
    | NoGrad var -> Printf.sprintf "nograd(%s)" (to_string var)
    | Or (lhs, rhs) -> Printf.sprintf "(%s | %s)" (to_string lhs) (to_string rhs)
    | Iota (index, shape) -> Printf.sprintf "iota(%d, %s)" index (String.concat "," (List.map string_of_int shape))
    | Reshape (var, shape) -> Printf.sprintf "reshape(%s, %s)" (to_string var) (String.concat "," (List.map string_of_int shape))
    | Sin var -> Printf.sprintf "sin(%s)" (to_string var)
    | Cos var -> Printf.sprintf "cos(%s)" (to_string var)
    | Concatenate (vars, axis) -> 
        Printf.sprintf "concat([%s], %d)" 
          (String.concat "; " (List.map to_string vars)) 
          axis
    | Select (cond, lhs, rhs) -> 
        Printf.sprintf "select(%s, %s, %s)" 
          (to_string cond) 
          (to_string lhs) 
          (to_string rhs)
    | Sqrt var -> Printf.sprintf "sqrt(%s)" (to_string var)
    | OptimizationBarrier var -> Printf.sprintf "barrier(%s)" (to_string var)

  module List = VarList
end

and ValueType : sig
  type ('a, 'b) u = shape * ('a, 'b) Tensor.kind

  module List : Hlist.S with type ('a, 'b) u = ('a, 'b) u

  type 'a t = 'a List.t

  val tensor_to_stable_hlo : ('a, 'b) u -> Stable_hlo.value_type

  val to_stable_hlo : 'a t -> Stable_hlo.value_type list

  val of_var : ('a, 'b) Var.u -> ('a, 'b) u

  val of_vars : 'a Var.t -> 'a t

  val of_tensor : ('a, 'b) Tensor.t -> ('a, 'b) u

  val to_arg : 'a t -> 'a Var.t
end = struct
  type ('a, 'b) u = shape * ('a, 'b) Tensor.kind

  module ValueTypeList : Hlist.S with type ('a, 'b) u = ('a, 'b) u =
  Hlist.Make (struct
    type ('a, 'b) t = ('a, 'b) u
  end)

  type 'a t = 'a ValueTypeList.t

  let tensor_to_stable_hlo (shape, tensor_element_type) =
    Stable_hlo.Tensor_type
      (shape, tensor_element_type_to_stable_hlo tensor_element_type)

  let to_stable_hlo l = ValueTypeList.map_to_list {f= tensor_to_stable_hlo} l

  let of_tensor : type a b. (a, b) Tensor.t -> (a, b) u =
   fun t ->
    let shape = Tensor.shape t in
    let kind = Tensor.kind t in
    ( shape
    , match kind with
      | F32 ->
          F32
      | I1 ->
          I1
      | I64 ->
          I64
      | U32 ->
          U32
      | U64 ->
          U64
      | F64 ->
          F64 )

  let rec of_var : type a b. (a, b) Var.u -> (a, b) u = function
    | Add (lhs, _) ->
        of_var lhs
    | Subtract (lhs, _) ->
        of_var lhs
    | Multiply (lhs, _) ->
        of_var lhs
    | Divide (lhs, _) ->
        of_var lhs
    | Abs var ->
        of_var var
    | Negate var ->
        of_var var
    | Ln var ->
        of_var var
    | Ln_1_plus var ->
        of_var var
    | Exponential var ->
        of_var var
    | Pow (lhs, _) ->
        of_var lhs
    | Argument (_, value_type) ->
        value_type
    | Compare (a, _, _) ->
        let shape, _ = of_var a in
        (shape, I1)
    | Min (lhs, _) ->
        of_var lhs
    | Max (lhs, _) ->
        of_var lhs
    | Constant t ->
        of_tensor t
    | BroadcastScalarConstant (value_type, _) ->
        value_type
    | DotProduct
        ( lhs
        , rhs
        , lhs_contracting_dims
        , rhs_contracting_dims
        , lhs_batching_dims
        , rhs_batching_dims ) ->
        let lhs_shape, element_type = of_var lhs in
        let rhs_shape, _ = of_var rhs in
        let batching_dims =
          List.map (fun i -> List.nth lhs_shape i) lhs_batching_dims
        in
        let lhs_remaining_dims =
          List.filteri
            (fun i _ ->
              not
                (List.mem i lhs_batching_dims || List.mem i lhs_contracting_dims) )
            lhs_shape
        in
        let rhs_remaining_dims =
          List.filteri
            (fun i _ ->
              not
                (List.mem i rhs_batching_dims || List.mem i rhs_contracting_dims) )
            rhs_shape
        in
        (batching_dims @ lhs_remaining_dims @ rhs_remaining_dims, element_type)
    | Random (value_type, _, _, _, _) ->
        value_type
    | DiffVar (_, v) ->
        of_var v
    | BroadcastInDim (var, new_dims) ->
        let old_dims, element_type = of_var var in
        (new_dims @ old_dims, element_type)
    | Transpose (var, permutation) ->
        let shape, element_type = of_var var in
        let new_shape = List.map (fun i -> List.nth shape i) permutation in
        (new_shape, element_type)
    | Tanh var ->
        of_var var
    | Sum (var, dimension) ->
        let shape, _ = of_var var in
        let new_shape =
          List.filteri (fun i _ -> not (List.mem i dimension)) shape
        in
        (new_shape, F32)
    | RightShift (lhs, _) ->
        of_var lhs
    | LeftShift (lhs, _) ->
        of_var lhs
    | Bitcast (var, new_type) ->
        let shape, _ = of_var var in
        (shape, new_type)
    | Convert (var, new_type) ->
        let shape, _ = of_var var in
        (shape, new_type)
    | NoGrad var ->
        of_var var
    | Or (lhs, _) ->
        of_var lhs
    | Iota (_, shape) ->
        (shape, U64)
    | Reshape (var, new_shape) ->
        let _, element_type = of_var var in
        (new_shape, element_type)
    | Sin var ->
        of_var var
    | Cos var ->
        of_var var
    | Concatenate (vars, axis) ->
        let vars = List.map of_var vars in
        let shape, element_type = List.hd vars in
        let new_shape =
          List.mapi
            (fun i _ ->
              if i = axis then
                List.fold_left
                  (fun acc (shape, _) -> acc + List.nth shape i)
                  0 vars
              else List.nth shape i )
            shape
        in
        (new_shape, element_type)
    | Select (_, lhs, _) ->
        of_var lhs
    | Sqrt var ->
        of_var var
    | OptimizationBarrier var ->
        of_var var

  let of_vars l =
    let open Hlist.Map (Var.List) (ValueTypeList) in
    map {f= of_var} l

  let to_arg : type a. a t -> a Var.t =
   fun l ->
    let open Hlist.Map (ValueTypeList) (Var.List) in
    map {f= (fun t -> Argument (new_id (), t))} l

  module List = ValueTypeList
end

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

let shape_of_var var = ValueType.of_var var |> function shape, _ -> shape

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
            Stable_hlo.value_type_to_string @@ ValueType.tensor_to_stable_hlo
            @@ ValueType.of_tensor tensor
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
            Stable_hlo.value_type_to_string @@ ValueType.tensor_to_stable_hlo
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
                        | Uniform ->
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
                    , let shape = shape_of_var var' in
                      if List.is_empty shape then "array<i64>"
                      else
                        "array<i64: "
                        ^ String.concat ","
                            (List.init
                               (List.length (shape_of_var var'))
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
              if List.length (shape_of_var var') = 0 then
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
            (output_id, ValueType.tensor_to_stable_hlo (shape, new_type))
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
            (output_id, ValueType.tensor_to_stable_hlo (shape, new_type))
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
    a ValueType.t -> (a Var.t -> b Var.t) -> (a, b) Func.t =
 fun inputs body ->
  let open Hlist.Map (ValueType.List) (Var.List) in
  let args = ValueType.to_arg inputs in
  let outputs = body args in
  let parameter_names = Var.get_args args in
  let parameter_names = List.map string_of_int parameter_names in
  {inputs; parameter_names; outputs; name= "main"}

let func_to_stable_hlo (func : ('a, 'b) Func.t) =
  let outputs, ops = vars_to_ops func.outputs in
  let ops = List.filter_map (fun x -> x) ops in
  let inputs = ValueType.to_stable_hlo func.inputs |> List.rev in
  let inputs = List.combine func.parameter_names inputs in
  let outputs = List.rev outputs in
  let return_ops = annotated_values_to_return_op outputs in
  let outputs = List.map snd outputs in
  Stable_hlo.{id= func.name; inputs; outputs; body= ops @ [return_ops]}

module StringMap = Map.Make (String)

let compile entry =
  Stable_hlo.func_to_string @@ func_to_stable_hlo entry
