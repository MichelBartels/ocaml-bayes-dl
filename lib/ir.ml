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
  | F32 : f32 tensor
  | I1 : i1 tensor
  | I64 : i64 tensor
  | U32 : u32 tensor
  | U64 : u64 tensor
  | F64 : f64 tensor

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
  type _ t =
    | Add : 'a tensor t * 'a tensor t -> 'a tensor t
    | Subtract : 'a tensor t * 'a tensor t -> 'a tensor t
    | Multiply : 'a tensor t * 'a tensor t -> 'a tensor t
    | Divide : 'a tensor t * 'a tensor t -> 'a tensor t
    | Abs : 'a tensor t -> 'a tensor t
    | Ln : 'a tensor t -> 'a tensor t
    | Exponential : 'a tensor t -> 'a tensor t
    | Pow : 'a tensor t * 'a tensor t -> 'a tensor t
    | Argument : id * 'a tensor ValueType.t -> 'a tensor t
    | Compare : 'a tensor t * comparison_direction * 'a tensor t -> i1 tensor t
    | Min : 'a tensor t * 'a tensor t -> 'a tensor t
    | Max : 'a tensor t * 'a tensor t -> 'a tensor t
    | Constant : 'a tensor ValueType.t * ('a, 'b) Tensor.t -> 'a tensor t
    | DotProduct :
        'a tensor t * 'a tensor t * int list * int list * int list * int list
        -> 'a tensor t
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
    | BroadcastInDim : 'a tensor t * int list -> 'a tensor t
    | Transpose : 'a tensor t * int list -> 'a tensor t
    | Tanh : 'a tensor t -> 'a tensor t
    | Sum : f32 tensor t * int list -> f32 tensor t
    | RightShift : u64 tensor t * u64 tensor t -> u64 tensor t
    | LeftShift : u64 tensor t * u64 tensor t -> u64 tensor t
    | Bitcast : 'a tensor t * 'b tensor -> 'b tensor t
    | Convert : 'a tensor t * 'b tensor -> 'b tensor t
    | NoGrad : 'a tensor t -> 'a tensor t
    | Or : u64 tensor t * u64 tensor t -> u64 tensor t
    | Iota : int * int list -> u64 tensor t
    | Reshape : 'a tensor t * int list -> 'a tensor t
    | Sin : 'a tensor t -> 'a tensor t
    | Cos : 'a tensor t -> 'a tensor t

  val to_var_list : 'a VarList.t t -> 'a VarList.t

  val from_var_list : 'a VarList.t -> 'a VarList.t t

  val to_annotated_values : 'a t -> (string * Stable_hlo.value_type) list

  val to_annotated_value : 'a tensor t -> string * Stable_hlo.value_type

  val length : 'a Var.t -> int

  val get_args : 'a Var.t -> id list

  type map2_fn = {fn: 'a. 'a tensor t -> 'a tensor t -> 'a tensor t}

  val map2 : map2_fn -> 'a t -> 'a t -> 'a t

  type 'b map_acc_fn = {fn: 'a. 'a tensor t -> 'b -> 'a tensor t * 'b}

  val map_acc : 'b map_acc_fn -> 'a t -> 'b -> 'a t * 'b

  type map_fn = {fn: 'a. 'a tensor t -> 'a tensor t}

  val map : map_fn -> 'a t -> 'a t
end = struct
  type _ t =
    | Add : 'a tensor t * 'a tensor t -> 'a tensor t
    | Subtract : 'a tensor t * 'a tensor t -> 'a tensor t
    | Multiply : 'a tensor t * 'a tensor t -> 'a tensor t
    | Divide : 'a tensor t * 'a tensor t -> 'a tensor t
    | Abs : 'a tensor t -> 'a tensor t
    | Ln : 'a tensor t -> 'a tensor t
    | Exponential : 'a tensor t -> 'a tensor t
    | Pow : 'a tensor t * 'a tensor t -> 'a tensor t
    | Argument : id * 'a tensor ValueType.t -> 'a tensor t
    | Compare : 'a tensor t * comparison_direction * 'a tensor t -> i1 tensor t
    | Min : 'a tensor t * 'a tensor t -> 'a tensor t
    | Max : 'a tensor t * 'a tensor t -> 'a tensor t
    | Constant : 'a tensor ValueType.t * ('a, 'b) Tensor.t -> 'a tensor t
    | DotProduct :
        'a tensor t * 'a tensor t * int list * int list * int list * int list
        -> 'a tensor t
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
    | BroadcastInDim : 'a tensor t * int list -> 'a tensor t
    | Transpose : 'a tensor t * int list -> 'a tensor t
    | Tanh : 'a tensor t -> 'a tensor t
    | Sum : f32 tensor t * int list -> f32 tensor t
    | RightShift : u64 tensor t * u64 tensor t -> u64 tensor t
    | LeftShift : u64 tensor t * u64 tensor t -> u64 tensor t
    | Bitcast : 'a tensor t * 'b tensor -> 'b tensor t
    | Convert : 'a tensor t * 'b tensor -> 'b tensor t
    | NoGrad : 'a tensor t -> 'a tensor t
    | Or : u64 tensor t * u64 tensor t -> u64 tensor t
    | Iota : int * int list -> u64 tensor t
    | Reshape : 'a tensor t * int list -> 'a tensor t
    | Sin : 'a tensor t -> 'a tensor t
    | Cos : 'a tensor t -> 'a tensor t

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

  let rec get_args : type a. a Var.t -> id list = function
    | Argument (id, _) ->
        [id]
    | [] ->
        []
    | x :: xs ->
        let l = Var.to_var_list (x :: xs) in
        VarList.fold_left {f= (fun args var -> get_args var @ args)} [] l
    | _ ->
        failwith "expected nested list of arguments"

  type map2_fn = {fn: 'a. 'a tensor t -> 'a tensor t -> 'a tensor t}

  type 'b map2_acc_fn =
    {fn: 'a. 'a tensor t -> 'a tensor t -> 'b -> 'a tensor t * 'b}

  let rec map2_acc : type a b. b map2_acc_fn -> a t -> a t -> b -> a t * b =
   fun {fn} a b acc ->
    match (a, b) with
    | (Add _ as a), b ->
        fn a b acc
    | (Subtract _ as a), b ->
        fn a b acc
    | (Multiply _ as a), b ->
        fn a b acc
    | (Divide _ as a), b ->
        fn a b acc
    | (Abs _ as a), b ->
        fn a b acc
    | Ln a, b ->
        fn a b acc
    | (Exponential _ as a), b ->
        fn a b acc
    | (Pow _ as a), b ->
        fn a b acc
    | (Argument _ as a), b ->
        fn a b acc
    | (Compare _ as a), b ->
        fn a b acc
    | (Min _ as a), b ->
        fn a b acc
    | (Max _ as a), b ->
        fn a b acc
    | (Constant _ as a), b ->
        fn a b acc
    | (DotProduct _ as a), b ->
        fn a b acc
    | (Random _ as a), b ->
        fn a b acc
    | (Transpose _ as a), b ->
        fn a b acc
    | [], [] ->
        ([], acc)
    | hd1 :: tl1, hd2 :: tl2 ->
        let tl, acc = map2_acc {fn} tl1 tl2 acc in
        let hd, acc = map2_acc {fn} hd1 hd2 acc in
        (hd :: tl, acc)
    | (DiffVar _ as a), b ->
        fn a b acc
    | (DiffConst _ as a), b ->
        fn a b acc
    | (BroadcastInDim _ as a), b ->
        fn a b acc
    | (Tanh _ as a), b ->
        fn a b acc
    | (Sum _ as a), b ->
        fn a b acc
    | (RightShift _ as a), b ->
        fn a b acc
    | (LeftShift _ as a), b ->
        fn a b acc
    | (Bitcast _ as a), b ->
        fn a b acc
    | (Convert _ as a), b ->
        fn a b acc
    | (NoGrad _ as a), b ->
        fn a b acc
    | (Or _ as a), b ->
        fn a b acc
    | (Iota _ as a), b ->
        fn a b acc
    | (Reshape _ as a), b ->
        fn a b acc
    | (Sin _ as a), b ->
        fn a b acc
    | (Cos _ as a), b ->
        fn a b acc

  let map2 ({fn} : map2_fn) a b =
    map2_acc {fn= (fun a b () -> (fn a b, ()))} a b () |> fst

  type 'b map_acc_fn = {fn: 'a. 'a tensor t -> 'b -> 'a tensor t * 'b}

  let map_acc ({fn} : 'a map_acc_fn) a acc =
    map2_acc {fn= (fun a _ acc -> fn a acc)} a a acc

  type map_fn = {fn: 'a. 'a tensor t -> 'a tensor t}

  let map ({fn} : map_fn) a = map2 {fn= (fun a _ -> fn a)} a a
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
        let (Tensor_type (shape, _)) = of_var a in
        Tensor_type (shape, I1)
    | Min (lhs, _) ->
        of_var lhs
    | Max (lhs, _) ->
        of_var lhs
    | Constant (value_type, _) ->
        value_type
    | DotProduct
        ( lhs
        , rhs
        , lhs_contracting_dims
        , rhs_contracting_dims
        , lhs_batching_dims
        , rhs_batching_dims ) ->
        let (Tensor_type (lhs_shape, element_type)) = of_var lhs in
        let (Tensor_type (rhs_shape, _)) = of_var rhs in
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
        Tensor_type
          (batching_dims @ lhs_remaining_dims @ rhs_remaining_dims, element_type)
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
    | BroadcastInDim (var, new_dims) ->
        let (Tensor_type (old_dims, element_type)) = of_var var in
        Tensor_type (new_dims @ old_dims, element_type)
    | Transpose (var, permutation) ->
        let (Tensor_type (shape, element_type)) = of_var var in
        let new_shape = List.map (fun i -> List.nth shape i) permutation in
        Tensor_type (new_shape, element_type)
    | Tanh var ->
        of_var var
    | Sum (var, dimension) ->
        let (Tensor_type (shape, _)) = of_var var in
        let new_shape =
          List.filteri (fun i _ -> not (List.mem i dimension)) shape
        in
        Tensor_type (new_shape, F32)
    | RightShift (lhs, _) ->
        of_var lhs
    | LeftShift (lhs, _) ->
        of_var lhs
    | Bitcast (var, new_type) ->
        let (Tensor_type (shape, _)) = of_var var in
        Tensor_type (shape, new_type)
    | Convert (var, new_type) ->
        let (Tensor_type (shape, _)) = of_var var in
        Tensor_type (shape, new_type)
    | NoGrad var ->
        of_var var
    | Or (lhs, _) ->
        of_var lhs
    | Iota (_, shape) ->
        Tensor_type (shape, U64)
    | Reshape (var, new_shape) ->
        let (Tensor_type (_, element_type)) = of_var var in
        Tensor_type (new_shape, element_type)
    | Sin var ->
        of_var var
    | Cos var ->
        of_var var

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

and Tensor : sig
  type ('a, 'b) t

  type (_, _) value =
    | F32 : float -> (f32, float) value
    | F64 : float -> (f64, float) value
    | I1 : bool -> (i1, bool) value
    | I64 : int -> (i64, int) value
    | U32 : string -> (u32, string) value
    | U64 : string -> (u64, string) value

  val full : ('a, 'b) value -> int list -> ('a, 'b) t

  val value_type : ('a, 'b) t -> 'a tensor ValueType.t

  val get : ('a, 'b) t -> int list -> ('a, 'b) value

  val shape : ('a, 'b) t -> shape

  val to_string : ('a, 'b) t -> string

  val to_ir : ('a, 'b) t -> 'a tensor Var.t

  val from_int_list : int list -> (i64, int) t

  val from_float_list : float list -> (f32, float) t

  val scalar_f32 : float -> (f32, float) t

  val scalar_u64 : string -> (u64, string) t
end = struct
  type _ tensor_values =
    | Full : 'a -> 'a tensor_values
    | List : 'a list -> 'a tensor_values

  type ('a, 'b) generic_tensor = int list * 'b tensor_values

  type (_, _) value =
    | F32 : float -> (f32, float) value
    | F64 : float -> (f64, float) value
    | I1 : bool -> (i1, bool) value
    | I64 : int -> (i64, int) value
    | U32 : string -> (u32, string) value
    | U64 : string -> (u64, string) value

  let value_to_string : type a b. (a, b) value -> string =
   fun v ->
    match v with
    | F32 f | F64 f ->
        Printf.sprintf "%e" f
    | I1 b ->
        string_of_bool b
    | I64 i ->
        string_of_int i
    | U32 i | U64 i ->
        i

  type (_, _) t =
    | F32 : (f32, float) generic_tensor -> (f32, float) t
    | F64 : (f64, float) generic_tensor -> (f64, float) t
    | I1 : (i1, bool) generic_tensor -> (i1, bool) t
    | I64 : (i64, int) generic_tensor -> (i64, int) t
    | U32 : (u32, string) generic_tensor -> (u32, string) t
    | U64 : (u64, string) generic_tensor -> (u64, string) t

  let full : type a b. (a, b) value -> int list -> (a, b) t =
   fun value shape ->
    match value with
    | F32 f ->
        F32 (shape, Full f)
    | F64 f ->
        F64 (shape, Full f)
    | I1 b ->
        I1 (shape, Full b)
    | I64 i ->
        I64 (shape, Full i)
    | U32 i ->
        U32 (shape, Full i)
    | U64 i ->
        U64 (shape, Full i)

  let value_type : type a b. (a, b) t -> a tensor ValueType.t =
   fun t ->
    match t with
    | F32 (shape, _) ->
        ValueType.Tensor_type (shape, F32)
    | F64 (shape, _) ->
        ValueType.Tensor_type (shape, F64)
    | I1 (shape, _) ->
        ValueType.Tensor_type (shape, I1)
    | I64 (shape, _) ->
        ValueType.Tensor_type (shape, I64)
    | U32 (shape, _) ->
        ValueType.Tensor_type (shape, U32)
    | U64 (shape, _) ->
        ValueType.Tensor_type (shape, U64)

  let calc_value : type a. a tensor_values -> int list -> a = function
    | Full v ->
        Fun.const v
    | List l ->
        fun i -> List.nth l (List.hd i)

  let get : type a b. (a, b) t -> int list -> (a, b) value =
   fun t idx ->
    match t with
    | F32 (_, f) ->
        F32 (calc_value f idx)
    | F64 (_, f) ->
        F64 (calc_value f idx)
    | I1 (_, b) ->
        I1 (calc_value b idx)
    | I64 (_, i) ->
        I64 (calc_value i idx)
    | U32 (_, i) ->
        U32 (calc_value i idx)
    | U64 (_, i) ->
        U64 (calc_value i idx)

  let shape : type a b. (a, b) t -> int list = function
    | F32 (shape, _)
    | F64 (shape, _)
    | I1 (shape, _)
    | I64 (shape, _)
    | U32 (shape, _)
    | U64 (shape, _) ->
        shape

  type 'a values = Tensor of 'a values Seq.t | Value of 'a

  let values t =
    let shape = shape t in
    let rec values' shape acc =
      match shape with
      | [] ->
          Value (get t acc)
      | x :: xs ->
          Tensor (Seq.init x (fun i -> values' xs (i :: acc)))
    in
    values' shape []

  let to_string t =
    let rec values_to_string = function
      | Tensor s ->
          "["
          ^ (Seq.map values_to_string s |> List.of_seq |> String.concat ", ")
          ^ "]"
      | Value v ->
          value_to_string v
    in
    let data = values_to_string (values t) in
    let signature =
      Stable_hlo.value_type_to_string
        (value_type t |> ValueType.tensor_to_stable_hlo)
    in
    Printf.sprintf "dense<%s> : %s" data signature

  let to_ir t =
    let value_type = value_type t in
    Var.Constant (value_type, t)

  let from_int_list l = I64 ([List.length l], List l)

  let from_float_list l = F32 ([List.length l], List l)

  let scalar_f32 f = F32 ([], Full f)

  let scalar_u64 i = U64 ([], Full i)
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
      | Constant (_, repr) ->
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
      | [] ->
          (prev_outputs, cache)
      | hd :: tl ->
          let hd, cache = aux ([], cache) hd in
          aux (hd @ prev_outputs, cache) tl
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
            aux ([], cache)
              (Constant (Tensor_type ([], F32), Tensor.scalar_f32 0.0))
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
            ( output_id
            , ValueType.tensor_to_stable_hlo (Tensor_type (shape, new_type)) )
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
            ( output_id
            , ValueType.tensor_to_stable_hlo (Tensor_type (shape, new_type)) )
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
  in
  let outputs, cache = aux ([], VarMap.empty) vars in
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
  let open Hlist.Map (ValueTypeList) (VarList) in
  let args = ValueType.to_arg inputs in
  let outputs = body args in
  let parameter_names = Var.get_args args in
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
    | Divide (a, b) ->
        all_funcs (all_funcs cache a) b
    | Abs a ->
        all_funcs cache a
    | Ln a ->
        all_funcs cache a
    | Exponential a ->
        all_funcs cache a
    | Pow (a, b) ->
        all_funcs (all_funcs cache a) b
    | Argument _ ->
        cache
    | Compare (a, _, b) ->
        all_funcs (all_funcs cache a) b
    | Min (a, b) ->
        all_funcs (all_funcs cache a) b
    | Max (a, b) ->
        all_funcs (all_funcs cache a) b
    | Constant _ ->
        cache
    | DotProduct (a, b, _, _, _, _) ->
        all_funcs (all_funcs cache a) b
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
    | BroadcastInDim (var, _) ->
        all_funcs cache var
    | Transpose (var, _) ->
        all_funcs cache var
    | Tanh var ->
        all_funcs cache var
    | Sum (var, _) ->
        all_funcs cache var
    | RightShift (a, b) ->
        all_funcs (all_funcs cache a) b
    | LeftShift (a, b) ->
        all_funcs (all_funcs cache a) b
    | Bitcast (a, _) ->
        all_funcs cache a
    | Convert (a, _) ->
        all_funcs cache a
    | NoGrad var ->
        all_funcs cache var
    | Or (a, b) ->
        all_funcs (all_funcs cache a) b
    | Iota _ ->
        cache
    | Reshape (a, _) ->
        all_funcs cache a
    | Sin a ->
        all_funcs cache a
    | Cos a ->
        all_funcs cache a
  in
  let main = func_to_stable_hlo entry |> Stable_hlo.func_to_string in
  let cache = StringMap.add entry.Func.name main StringMap.empty in
  let funcs = all_funcs cache entry.Func.outputs in
  StringMap.bindings funcs |> List.map snd |> String.concat "\n"
