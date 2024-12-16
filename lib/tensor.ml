type ('a, 'b) generic_tensor = int list * (int list -> 'b)

type (_, _) value =
  | F32 : float -> (Ir.f32, float) value
  | I1 : bool -> (Ir.i1, bool) value
  | I64 : int -> (Ir.i64, int) value

let value_to_string : type a b. (a, b) value -> string =
 fun v ->
  match v with
  | F32 f ->
      Printf.sprintf "%f10" f
  | I1 b ->
      string_of_bool b
  | I64 i ->
      string_of_int i

type (_, _) t =
  | F32 : (Ir.f32, float) generic_tensor -> (Ir.f32, float) t
  | I1 : (Ir.i1, bool) generic_tensor -> (Ir.i1, bool) t
  | I64 : (Ir.i64, int) generic_tensor -> (Ir.i64, int) t

let full : type a b. (a, b) value -> int list -> (a, b) t =
 fun value shape ->
  match value with
  | F32 f ->
      F32 (shape, fun _ -> f)
  | I1 b ->
      I1 (shape, fun _ -> b)
  | I64 i ->
      I64 (shape, fun _ -> i)

let value_type : type a b. (a, b) t -> a Ir.tensor Ir.ValueType.t =
 fun t ->
  match t with
  | F32 (shape, _) ->
      Ir.ValueType.Tensor_type (shape, F32)
  | I1 (shape, _) ->
      Ir.ValueType.Tensor_type (shape, I1)
  | I64 (shape, _) ->
      Ir.ValueType.Tensor_type (shape, I64)

let get : type a b. (a, b) t -> int list -> (a, b) value =
 fun t idx ->
  match t with
  | F32 (_, f) ->
      F32 (f idx)
  | I1 (_, b) ->
      I1 (b idx)
  | I64 (_, i) ->
      I64 (i idx)

let shape : type a b. (a, b) t -> int list =
 fun t ->
  match t with
  | F32 (shape, _) ->
      shape
  | I1 (shape, _) ->
      shape
  | I64 (shape, _) ->
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
        ^ Seq.fold_lefti
            (fun acc i x ->
              (if i = 0 then "" else acc ^ ", ") ^ values_to_string x )
            "" s
        ^ "]"
    | Value v ->
        value_to_string v
  in
  let data = values_to_string (values t) in
  let signature =
    Stable_hlo.value_type_to_string
      (value_type t |> Ir.ValueType.tensor_to_stable_hlo)
  in
  Printf.sprintf "dense<%s> : %s" data signature

let to_ir t =
  let repr = to_string t in
  let value_type = value_type t in
  Ir.Var.Constant (value_type, repr)

let from_int_list l = I64 ([List.length l], fun i -> List.nth l (List.hd i))

let from_float_list l = F32 ([List.length l], fun i -> List.nth l (List.hd i))

let scalar_f32 f = F32 ([], fun _ -> f)
