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
type shape = int list
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

  module List = ValueTypeList
