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

type u32 = U32

type u64 = U64

type f64 = F64

type _ tensor =
  | F32 : (f32 * float) tensor
  | F64 : (f64 * float) tensor
  | I1 : (i1 * bool) tensor
  | I64 : (i64 * Signed.Int64.t) tensor
  | U32 : (u32 * Unsigned.uint32) tensor
  | U64 : (u64 * Unsigned.uint64) tensor

let zeros : type a b. (a * b) tensor -> b = function
  | F32 ->
      0.0
  | F64 ->
      0.0
  | I1 ->
      false
  | I64 ->
      Signed.Int64.zero
  | U32 ->
      Unsigned.UInt32.zero
  | U64 ->
      Unsigned.UInt64.zero

let ones : type a b. (a * b) tensor -> b = function
  | F32 ->
      1.0
  | F64 ->
      1.0
  | I1 ->
      true
  | I64 ->
      Signed.Int64.one
  | U32 ->
      Unsigned.UInt32.one
  | U64 ->
      Unsigned.UInt64.one

type shape = int list

let tensor_element_type_to_stable_hlo :
    type a. a tensor -> Stable_hlo.tensor_element_type = function
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
  type _ u =
    | Add : 'a u * 'a u -> 'a u
    | Subtract : 'a u * 'a u -> 'a u
    | Multiply : 'a u * 'a u -> 'a u
    | Divide : 'a u * 'a u -> 'a u
    | Abs : 'a u -> 'a u
    | Ln : 'a u -> 'a u
    | Exponential : 'a u -> 'a u
    | Pow : 'a u * 'a u -> 'a u
    | Argument : id * 'a ValueType.u -> 'a u
    | Compare : 'a u * comparison_direction * 'a u -> (i1 * bool) u
    | Min : 'a u * 'a u -> 'a u
    | Max : 'a u * 'a u -> 'a u
    | Constant : ('a * 'b) Tensor.t -> ('a * 'b) u
    | DotProduct :
        'a u * 'a u * int list * int list * int list * int list
        -> 'a u
    | Random :
        'a ValueType.u
        * (f32 * float) u
        * (f32 * float) u
        * (i64 * Signed.Int64.t) u
        * distribution
        -> 'a u
    | DiffVar : id * 'a u -> 'a u
    | DiffConst : 'a u -> 'a u
    | BroadcastInDim : 'a u * int list -> 'a u
    | Transpose : 'a u * int list -> 'a u
    | Tanh : 'a u -> 'a u
    | Sum : (f32 * float) u * int list -> (f32 * float) u
    | RightShift :
        (u64 * Unsigned.uint64) u * (u64 * Unsigned.uint64) u
        -> (u64 * Unsigned.uint64) u
    | LeftShift :
        (u64 * Unsigned.uint64) u * (u64 * Unsigned.uint64) u
        -> (u64 * Unsigned.uint64) u
    | Bitcast : 'a u * ('c * 'd) tensor -> ('c * 'd) u
    | Convert : 'a u * ('c * 'd) tensor -> ('c * 'd) u
    | NoGrad : 'a u -> 'a u
    | Or :
        (u64 * Unsigned.uint64) u * (u64 * Unsigned.uint64) u
        -> (u64 * Unsigned.uint64) u
    | Iota : int * int list -> (u64 * Unsigned.uint64) u
    | Reshape : 'a u * int list -> 'a u
    | Sin : 'a u -> 'a u
    | Cos : 'a u -> 'a u
    | Concatenate : 'a u list * int -> 'a u

  module List : Hlist.S with type 'a u = 'a u and type 'a v = 'a u

  type 'a t = 'a List.t

  val to_annotated_values : 'a t -> (string * Stable_hlo.value_type) list

  val to_annotated_value : 'a u -> string * Stable_hlo.value_type

  val length : 'a Var.t -> int

  val get_args : 'a Var.t -> id list

  type map2_fn = {f: 'a. 'a u -> 'a u -> 'a u}

  val map2 : map2_fn -> 'a t -> 'a t -> 'a t

  type 'a map2_acc_fn = 'a List.map2_acc_fn

  type 'b map_acc_fn = {f: 'a. 'a u -> 'b -> 'a u * 'b}

  val map_acc : 'b map_acc_fn -> 'a t -> 'b -> 'a t * 'b

  type map_fn = {f: 'a. 'a u -> 'a u}

  val map : map_fn -> 'a t -> 'a t
end = struct
  type _ u =
    | Add : 'a u * 'a u -> 'a u
    | Subtract : 'a u * 'a u -> 'a u
    | Multiply : 'a u * 'a u -> 'a u
    | Divide : 'a u * 'a u -> 'a u
    | Abs : 'a u -> 'a u
    | Ln : 'a u -> 'a u
    | Exponential : 'a u -> 'a u
    | Pow : 'a u * 'a u -> 'a u
    | Argument : id * 'a ValueType.u -> 'a u
    | Compare : 'a u * comparison_direction * 'a u -> (i1 * bool) u
    | Min : 'a u * 'a u -> 'a u
    | Max : 'a u * 'a u -> 'a u
    | Constant : ('a * 'b) Tensor.t -> ('a * 'b) u
    | DotProduct :
        'a u * 'a u * int list * int list * int list * int list
        -> 'a u
    | Random :
        'a ValueType.u
        * (f32 * float) u
        * (f32 * float) u
        * (i64 * Signed.Int64.t) u
        * distribution
        -> 'a u
    | DiffVar : id * 'a u -> 'a u
    | DiffConst : 'a u -> 'a u
    | BroadcastInDim : 'a u * int list -> 'a u
    | Transpose : 'a u * int list -> 'a u
    | Tanh : 'a u -> 'a u
    | Sum : (f32 * float) u * int list -> (f32 * float) u
    | RightShift :
        (u64 * Unsigned.uint64) u * (u64 * Unsigned.uint64) u
        -> (u64 * Unsigned.uint64) u
    | LeftShift :
        (u64 * Unsigned.uint64) u * (u64 * Unsigned.uint64) u
        -> (u64 * Unsigned.uint64) u
    | Bitcast : 'a u * ('c * 'd) tensor -> ('c * 'd) u
    | Convert : 'a u * ('c * 'd) tensor -> ('c * 'd) u
    | NoGrad : 'a u -> 'a u
    | Or :
        (u64 * Unsigned.uint64) u * (u64 * Unsigned.uint64) u
        -> (u64 * Unsigned.uint64) u
    | Iota : int * int list -> (u64 * Unsigned.uint64) u
    | Reshape : 'a u * int list -> 'a u
    | Sin : 'a u -> 'a u
    | Cos : 'a u -> 'a u
    | Concatenate : 'a u list * int -> 'a u

  module VarList : Hlist.S with type 'a u = 'a u and type 'a v = 'a u =
  Hlist.Make (struct
    type 'a t = 'a u

    type 'a tag = 'a u
  end)

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

  type map2_fn = {f: 'a. 'a u -> 'a u -> 'a u}

  type 'a map2_acc_fn = 'a VarList.map2_acc_fn

  let map2_acc = VarList.map2_acc

  let map2 ({f} : map2_fn) a b =
    map2_acc {f= (fun a b () -> (f a b, ()))} a b () |> fst

  type 'b map_acc_fn = {f: 'a. 'a u -> 'b -> 'a u * 'b}

  let map_acc ({f} : 'a map_acc_fn) a acc =
    map2_acc {f= (fun a _ acc -> f a acc)} a a acc

  type map_fn = {f: 'a. 'a u -> 'a u}

  let map ({f} : map_fn) a = map2 {f= (fun a _ -> f a)} a a

  module List = VarList
end

and ValueType : sig
  type 'a u = shape * 'a tensor

  module List : Hlist.S with type 'a u = 'a u and type 'a v = 'a Var.u

  type 'a t = 'a List.t

  val tensor_to_stable_hlo : 'a u -> Stable_hlo.value_type

  val to_stable_hlo : 'a t -> Stable_hlo.value_type list

  val of_var : 'a Var.u -> 'a u

  val of_vars : 'a Var.t -> 'a t

  val to_arg : 'a t -> 'a Var.t
end = struct
  type 'a u = shape * 'a tensor

  module ValueTypeList :
    Hlist.S with type 'a u = 'a u and type 'a v = 'a Var.u = Hlist.Make (struct
    type 'a t = 'a u

    type 'a tag = 'a Var.u
  end)

  type 'a t = 'a ValueTypeList.t

  let tensor_to_stable_hlo (shape, tensor_element_type) =
    Stable_hlo.Tensor_type
      (shape, tensor_element_type_to_stable_hlo tensor_element_type)

  let to_stable_hlo l = ValueTypeList.map_to_list {f= tensor_to_stable_hlo} l

  let rec of_var : type a. a Var.u -> a u = function
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
    | Ln var ->
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
        Tensor.value_type t
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
                (List.mem i lhs_batching_dims || List.mem i lhs_contracting_dims)
              )
            lhs_shape
        in
        let rhs_remaining_dims =
          List.filteri
            (fun i _ ->
              not
                (List.mem i rhs_batching_dims || List.mem i rhs_contracting_dims)
              )
            rhs_shape
        in
        (batching_dims @ lhs_remaining_dims @ rhs_remaining_dims, element_type)
    | Random (value_type, _, _, _, _) ->
        value_type
    | DiffVar (_, v) ->
        of_var v
    | DiffConst v ->
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

and Tensor : sig
  type 'a t

  val c_type : ('a * 'b) tensor -> 'b Ctypes.typ

  type any_typ = Any : 'a Ctypes.typ -> any_typ

  val any_c_type : 'a tensor -> any_typ

  val kind : 'a t -> 'a tensor

  val full : ('a * 'b) tensor -> 'b -> int list -> ('a * 'b) t

  val value_type : 'a t -> 'a ValueType.u

  val get : ('a * 'b) t -> int list -> 'b

  val shape : ('a * 'b) t -> shape

  val to_string : ('a * 'b) t -> string

  val to_ir : ('a * 'b) t -> ('a * 'b) Var.u

  val from_int_list :
    ?shape:shape -> Signed.Int64.t list -> (i64 * Signed.Int64.t) t

  val from_float_list : ?shape:shape -> float list -> (f32 * float) t

  val from_carray :
    ('a * 'b) tensor -> shape -> 'b Ctypes.CArray.t -> ('a * 'b) t

  val scalar_f32 : float -> (f32 * float) t

  val scalar_u64 : string -> (u64 * Unsigned.uint64) t

  val carray : ('a * 'b) t -> 'b Ctypes.CArray.t

  type any_carray = Any : 'a Ctypes.CArray.t -> any_carray

  val any_carray : 'a t -> any_carray

  val to_scalar : ('a * 'b) t -> 'b

  val zeros : 'a tensor -> int list -> 'a t
end = struct
  open Ctypes

  let value_to_string : type a b. (a * b) tensor -> b -> string =
   fun kind v ->
    match (kind, v) with
    | F32, v ->
        Printf.sprintf "%e" v
    | F64, v ->
        Printf.sprintf "%e" v
    | I1, b ->
        string_of_bool b
    | I64, i ->
        Signed.Int64.to_string i
    | U32, i ->
        Unsigned.UInt32.to_string i
    | U64, i ->
        Unsigned.UInt64.to_string i

  type ('a, 'b) t' =
    {kind: ('a * 'b) tensor; arr: 'b Ctypes.CArray.t; shape: int list}

  type 'a t = Tensor : ('a, 'b) t' -> ('a * 'b) t

  let kind : type a. a t -> a tensor = fun (Tensor {kind; _}) -> kind

  type any_typ = Any : 'a typ -> any_typ

  let c_type : type a b. (a * b) tensor -> b typ = function
    | F32 ->
        float
    | F64 ->
        double
    | I1 ->
        bool
    | I64 ->
        int64_t
    | U32 ->
        uint32_t
    | U64 ->
        uint64_t

  let any_c_type : type a. a tensor -> any_typ = function
    | F32 ->
        Any float
    | F64 ->
        Any double
    | I1 ->
        Any bool
    | I64 ->
        Any int64_t
    | U32 ->
        Any uint32_t
    | U64 ->
        Any uint64_t

  type any_carray = Any : 'a Ctypes.CArray.t -> any_carray

  let any_carray : type a. a t -> any_carray = function
    | Tensor {arr; _} ->
        Any arr

  let full : type a b. (a * b) tensor -> b -> int list -> (a * b) t =
   fun kind initial shape ->
    let size = List.fold_left ( * ) 1 shape in
    Tensor {kind; arr= CArray.make ~initial (c_type kind) size; shape}

  let zeros : type a. a tensor -> int list -> a t =
   fun kind shape ->
    match kind with
    | F32 ->
        full kind 0.0 shape
    | F64 ->
        full kind 0.0 shape
    | I1 ->
        full kind false shape
    | I64 ->
        full kind Signed.Int64.zero shape
    | U32 ->
        full kind Unsigned.UInt32.zero shape
    | U64 ->
        full kind Unsigned.UInt64.zero shape

  let value_type : type a. a t -> a ValueType.u =
   fun (Tensor {kind; shape; _}) ->
    match kind with
    | F32 ->
        (shape, F32)
    | F64 ->
        (shape, F64)
    | I1 ->
        (shape, I1)
    | I64 ->
        (shape, I64)
    | U32 ->
        (shape, U32)
    | U64 ->
        (shape, U64)

  let calc_index = List.fold_left2 (fun acc i j -> (acc * i) + j) 0

  let get : type a b. (a * b) t -> int list -> b =
   fun (Tensor t) idx -> CArray.get t.arr (calc_index t.shape idx)

  let shape (Tensor {shape; _}) = shape

  type 'a values = Tensor of 'a values Seq.t | Value of 'a

  let values t =
    let shape = shape t in
    let rec values' shape acc =
      match shape with
      | [] ->
          Value (get t (List.rev acc))
      | x :: xs ->
          Tensor (Seq.init x (fun i -> values' xs (i :: acc)))
    in
    values' shape []

  let to_string (Tensor t : 'a t) =
    let rec values_to_string = function
      | Tensor s ->
          "["
          ^ (Seq.map values_to_string s |> List.of_seq |> String.concat ", ")
          ^ "]"
      | Value v ->
          value_to_string t.kind v
    in
    let data = values_to_string (values (Tensor t)) in
    let signature =
      Stable_hlo.value_type_to_string
        (value_type (Tensor t) |> ValueType.tensor_to_stable_hlo)
    in
    Printf.sprintf "dense<%s> : %s" data signature

  let to_ir t = Var.Constant t

  let from_list ?shape (kind : ('a * 'b) tensor) list : ('a * 'b) t =
    Tensor
      { kind
      ; arr= CArray.of_list (c_type kind) list
      ; shape= Option.value shape ~default:[List.length list] }

  let from_int_list ?shape = from_list ?shape I64

  let from_float_list ?shape = from_list ?shape F32

  let scalar_f32 f = full F32 f []

  let scalar_u64 s = full U64 (Unsigned.UInt64.of_string s) []

  let carray (Tensor {arr; _} : ('a * 'b) t) = arr

  let from_carray (kind : ('a * 'b) tensor) shape arr : ('a * 'b) t =
    Tensor {kind; arr; shape}

  let to_scalar t = get t (List.map (fun _ -> 0) (shape t))
end

let shape_of_var var = ValueType.of_var var |> function shape, _ -> shape

module AnnotatedValueSet = Set.Make (struct
  type t = string * Stable_hlo.value_type

  let compare = Stdlib.compare
end)

type any_var = Any_var : 'a Var.u -> any_var

module VarMap = struct
  type 'a t = (any_var * 'a) list

  let empty = []

  let add var value map =
    if List.mem_assoc var map then map else (var, value) :: map

  let mem = List.mem_assoc

  let find = List.assoc

  let bindings map = map
end

let vars_to_ops vars =
  let rec aux :
      type a.
         Stable_hlo.annotated_value list
         * (Stable_hlo.op option * Stable_hlo.annotated_value) VarMap.t
      -> a Var.u
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
              ; anonymous_functions= []
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
              ; anonymous_functions= []
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
              ; anonymous_functions= []
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Constant repr ->
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= []
              ; outputs= [output]
              ; name= "stablehlo.constant"
              ; attributes= [("value", Tensor.to_string repr)]
              ; anonymous_functions= []
              ; call= false }
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
              ; anonymous_functions= []
              ; call= false }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | DiffVar (_, var) ->
          aux (prev_outputs, cache) var
      | DiffConst var ->
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
              ; call= false }
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
                ; call= false }
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
              ; call= false }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
      | Sum (var', dimensions) ->
          let var', cache = aux ([], cache) var' in
          let initial, cache =
            aux ([], cache) (Constant (Tensor.scalar_f32 0.0))
          in
          let output = Var.to_annotated_value var in
          let op =
            Stable_hlo.
              { inputs= var' @ initial
              ; outputs= [output]
              ; name= "stablehlo.reduce"
              ; attributes=
                  [ ( "dimensions"
                    , "array<i64: "
                      ^ String.concat "," (List.map string_of_int dimensions)
                      ^ ">" ) ]
              ; anonymous_functions=
                  (let scalar_type = Tensor_type ([], F32) in
                   [ Stable_hlo.
                       { id= "sum"
                       ; inputs= [("x", scalar_type); ("y", scalar_type)]
                       ; outputs= [scalar_type]
                       ; body=
                           [ { inputs= [("x", scalar_type); ("y", scalar_type)]
                             ; outputs= [("z", scalar_type)]
                             ; name= "stablehlo.add"
                             ; attributes= []
                             ; anonymous_functions= []
                             ; call= false }
                           ; { inputs= [("z", scalar_type)]
                             ; outputs= []
                             ; name= "stablehlo.return"
                             ; attributes= []
                             ; anonymous_functions= []
                             ; call= false } ] } ] )
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
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
              ; call= false }
          in
          (output :: prev_outputs, add var (Some op, output) cache)
  in
  let outputs, cache = Var.List.fold_left {f= aux} ([], VarMap.empty) vars in
  (outputs, VarMap.bindings cache |> List.map snd |> List.map fst |> List.rev)

let annotated_values_to_return_op values =
  Stable_hlo.
    { inputs= values
    ; outputs= []
    ; name= "stablehlo.return"
    ; attributes= []
    ; anonymous_functions= []
    ; call= false }

let create_func :
    type a b. a ValueType.t -> (a Var.t -> b Var.t) -> (a, b) Func.t =
 fun inputs body ->
  let open Hlist.Map (ValueType.List) (Var.List) in
  let args = ValueType.to_arg inputs in
  let outputs = body args in
  let parameter_names = Var.get_args args in
  let parameter_names = List.map string_of_int parameter_names in
  {inputs; parameter_names; outputs; name= "fn" ^ string_of_int (new_id ())}

let func_to_stable_hlo (func : ('a, 'b) Func.t) =
  let outputs, ops = vars_to_ops func.outputs in
  let ops = List.filter_map (fun x -> x) ops in
  let inputs = ValueType.to_stable_hlo func.inputs in
  let inputs = List.combine func.parameter_names inputs |> List.rev in
  let outputs = List.rev outputs in
  let return_ops = annotated_values_to_return_op outputs in
  let outputs = List.map snd outputs in
  Stable_hlo.{id= func.name; inputs; outputs; body= ops @ [return_ops]}

module StringMap = Map.Make (String)

let compile entry =
  (* let rec all_funcs : *)
  (*     type a. string StringMap.t -> a Var.t -> string StringMap.t = *)
  (*  fun cache var -> *)
  (*   match var with *)
  (*   | Var.Add (a, b) -> *)
  (*       print_endline "Add" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Subtract (a, b) -> *)
  (*       print_endline "Subtract" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Multiply (a, b) -> *)
  (*       print_endline "Multiply" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Divide (a, b) -> *)
  (*       print_endline "Divide" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Abs a -> *)
  (*       print_endline "Abs" ; all_funcs cache a *)
  (*   | Ln a -> *)
  (*       print_endline "Ln" ; all_funcs cache a *)
  (*   | Exponential a -> *)
  (*       print_endline "Exponential" ; *)
  (*       all_funcs cache a *)
  (*   | Pow (a, b) -> *)
  (*       print_endline "Pow" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Argument _ -> *)
  (*       print_endline "Argument" ; cache *)
  (*   | Compare (a, _, b) -> *)
  (*       print_endline "Compare" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Min (a, b) -> *)
  (*       print_endline "Min" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Max (a, b) -> *)
  (*       print_endline "Max" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Constant _ -> *)
  (*       print_endline "Constant" ; cache *)
  (*   | DotProduct (a, b, _, _, _, _) -> *)
  (*       print_endline "DotProduct" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Random (_, a, b, c, _) -> *)
  (*       print_endline "Random" ; *)
  (*       all_funcs (all_funcs (all_funcs cache a) b) c *)
  (*   | [] -> *)
  (*       print_endline "Empty" ; cache *)
  (*   | x :: xs -> *)
  (*       print_endline "List" ; *)
  (*       let l = Var.to_var_list (x :: xs) in *)
  (*       VarList.fold_left {f= all_funcs} cache l *)
  (*   | DiffVar (_, var) -> *)
  (*       print_endline "DiffVar" ; all_funcs cache var *)
  (*   | DiffConst var -> *)
  (*       print_endline "DiffConst" ; all_funcs cache var *)
  (*   | BroadcastInDim (var, _) -> *)
  (*       print_endline "BroadcastInDim" ; *)
  (*       all_funcs cache var *)
  (*   | Transpose (var, _) -> *)
  (*       print_endline "Transpose" ; all_funcs cache var *)
  (*   | Tanh var -> *)
  (*       print_endline "Tanh" ; all_funcs cache var *)
  (*   | Sum (var, _) -> *)
  (*       print_endline "Sum" ; all_funcs cache var *)
  (*   | RightShift (a, b) -> *)
  (*       print_endline "RightShift" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | LeftShift (a, b) -> *)
  (*       print_endline "LeftShift" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Bitcast (a, _) -> *)
  (*       print_endline "Bitcast" ; all_funcs cache a *)
  (*   | Convert (a, _) -> *)
  (*       print_endline "Convert" ; all_funcs cache a *)
  (*   | NoGrad var -> *)
  (*       print_endline "NoGrad" ; all_funcs cache var *)
  (*   | Or (a, b) -> *)
  (*       print_endline "Or" ; *)
  (*       all_funcs (all_funcs cache a) b *)
  (*   | Iota _ -> *)
  (*       print_endline "Iota" ; cache *)
  (*   | Reshape (a, _) -> *)
  (*       print_endline "Reshape" ; all_funcs cache a *)
  (*   | Sin a -> *)
  (*       print_endline "Sin" ; all_funcs cache a *)
  (*   | Cos a -> *)
  (*       print_endline "Cos" ; all_funcs cache a *)
  (*   | Concatenate (vars, _) -> *)
  (*       print_endline "Concatenate" ; *)
  (*       List.fold_left (fun acc var -> all_funcs acc var) cache vars *)
  (* in *)
  Stable_hlo.func_to_string @@ func_to_stable_hlo entry
(* let main = func_to_stable_hlo entry |> Stable_hlo.func_to_string in *)
(* print_endline "Generated hlo" ; *)
(* let cache = StringMap.add entry.Func.name main StringMap.empty in *)
(* print_endline "Added main to cache" ; *)
(* let funcs = all_funcs cache entry.Func.outputs in *)
(* print_endline "Got all funcs" ; *)
(* StringMap.bindings funcs |> List.map snd |> String.concat "\n" *)
