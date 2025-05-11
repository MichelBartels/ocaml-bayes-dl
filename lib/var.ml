module Tensor = Device_api.Tensor

type id = int

type 'a tagged = id * 'a
type comparison_direction = Eq | Ne | Ge | Gt | Le | Lt

type distribution = Uniform | Normal

let current_id = ref 0

let new_id () =
  let id = !current_id in
  current_id := id + 1 ;
  id

let tag x = (string_of_int @@ new_id (), x)

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
    | Argument : id * ('a, 'b) Value_type.u -> ('a, 'b) u
    | Compare :
        ('a, 'b) u * comparison_direction * ('a, 'b) u
        -> (Tensor.i1, bool) u
    | Min : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Max : ('a, 'b) u * ('a, 'b) u -> ('a, 'b) u
    | Constant : ('a, 'b) Tensor.t -> ('a, 'b) u
    | BroadcastScalarConstant : ('a, 'b) Value_type.u * 'b -> ('a, 'b) u
    | DotProduct :
        ('a, 'b) u * ('a, 'b) u * int list * int list * int list * int list
        -> ('a, 'b) u
    | Random :
        ('a, 'b) Value_type.u
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

  let rec value_type : type a b. (a, b) u -> (a, b) Value_type.u = function
    | Add (lhs, _) ->
        value_type lhs
    | Subtract (lhs, _) ->
        value_type lhs
    | Multiply (lhs, _) ->
        value_type lhs
    | Divide (lhs, _) ->
        value_type lhs
    | Abs var ->
        value_type var
    | Negate var ->
        value_type var
    | Ln var ->
        value_type var
    | Ln_1_plus var ->
        value_type var
    | Exponential var ->
        value_type var
    | Pow (lhs, _) ->
        value_type lhs
    | Argument (_, value_type) ->
        value_type
    | Compare (a, _, _) ->
        let shape, _ = value_type a in
        (shape, I1)
    | Min (lhs, _) ->
        value_type lhs
    | Max (lhs, _) ->
        value_type lhs
    | Constant t ->
        Value_type.of_tensor t
    | BroadcastScalarConstant (value_type, _) ->
        value_type
    | DotProduct
        ( lhs
        , rhs
        , lhs_contracting_dims
        , rhs_contracting_dims
        , lhs_batching_dims
        , rhs_batching_dims ) ->
        let lhs_shape, element_type = value_type lhs in
        let rhs_shape, _ = value_type rhs in
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
        value_type v
    | BroadcastInDim (var', new_dims) ->
        let old_dims, element_type = value_type var' in
        (new_dims @ old_dims, element_type)
    | Transpose (var', permutation) ->
        let shape, element_type = value_type var' in
        let new_shape = List.map (fun i -> List.nth shape i) permutation in
        (new_shape, element_type)
    | Tanh var ->
        value_type var
    | Sum (var, dimension) ->
        let shape, _ = value_type var in
        let new_shape =
          List.filteri (fun i _ -> not (List.mem i dimension)) shape
        in
        (new_shape, F32)
    | RightShift (lhs, _) ->
        value_type lhs
    | LeftShift (lhs, _) ->
        value_type lhs
    | Bitcast (var, new_type) ->
        let shape, _ = value_type var in
        (shape, new_type)
    | Convert (var, new_type) ->
        let shape, _ = value_type var in
        (shape, new_type)
    | NoGrad var ->
        value_type var
    | Or (lhs, _) ->
        value_type lhs
    | Iota (_, shape) ->
        (shape, U64)
    | Reshape (var, new_shape) ->
        let _, element_type = value_type var in
        (new_shape, element_type)
    | Sin var ->
        value_type var
    | Cos var ->
        value_type var
    | Concatenate (vars, axis) ->
        let vars = List.map value_type vars in
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
        value_type lhs
    | Sqrt var ->
        value_type var
    | OptimizationBarrier var ->
        value_type var

  let value_types l =
    let open Hlist.Map (VarList) (Value_type.List) in
    map {f= value_type} l

  let to_annotated_values var =
    List.map tag (Value_type.to_stable_hlo @@ value_types var)

  let to_annotated_value var =
    match var with
    | Argument (id, value_type) ->
        (string_of_int id, Value_type.tensor_to_stable_hlo value_type)
    | _ ->
        tag @@ Value_type.tensor_to_stable_hlo @@ value_type var

  let rec length : type a. a t -> int = function
    | [] ->
        0
    | x :: xs ->
        length x + length xs
    | _ ->
        1

  let rec get_args : type a. a t -> id list = function
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
  let arg_of_value_type l =
    let open Hlist.Map (Value_type.List) (VarList) in
    map {f= (fun t -> Argument (new_id (), t))} l

  module List = VarList

  let shape var =
    let shape, _ = value_type var in
    shape