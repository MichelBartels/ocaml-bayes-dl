type ('a, 'b, 'c, 'd, 'e) t =
  ('a Ir.VarList.t Ir.Var.t -> 'b Ir.Var.t)
  * 'a Ir.VarList.t Ir.ValueType.t
  * ('a Ir.VarList.t, 'e, 'c, 'd) Backpropagate.input

let return :
    type a b c. a Ir.Var.t -> (unit, a, b, unit Ir.VarList.t Ir.Var.t -> b, c) t
    =
 fun x -> ((fun _ -> x), List_type [], [])

let bind :
    type a b c d e f g h.
       (f, g, h, c Ir.VarList.t Ir.Var.t -> h, e) t
    -> (g Ir.Var.t -> (a, b, c, d, e) t)
    -> (a Ir.VarList.t Ir.Var.t -> f, b, h, d Ir.VarList.t Ir.Var.t -> h, e) t =
 fun (f, List_type t, b) g ->
  let _, t', b' = g (f (Ir.ValueType.to_arg (List_type t))) in
  ( (fun (x :: xs) ->
      let y = f xs in
      let z, _, _ = g y in
      z x )
  , List_type (t' :: t)
  , b' :: b )

let ( let* ) = bind

let new_param :
    type a b c.
       a Ir.tensor Ir.ValueType.t
    -> ( a Ir.tensor Ir.Var.t -> unit
       , a Ir.tensor
       , b
       , (c Ir.Var.t -> unit) Ir.VarList.t Ir.Var.t -> b
       , c )
       t =
 fun t -> ((fun [x] -> x), List_type [t], [Var])

let grad_and_value (f, t, b) =
  ((fun p -> Ir.Var.(p :: Backpropagate.diff b f p)), t, b)

let create_func :
    type a b c d e f.
       a Ir.ValueType.t
    -> (a Ir.Var.t -> (b, c, d, e, f) t)
    -> ( (a Ir.Var.t -> b Ir.VarList.t Ir.Var.t -> unit) Ir.VarList.t
       , c )
       Ir.Func.t =
 fun t f ->
  let args = Ir.ValueType.to_arg t in
  let _, t', _ = f args in
  Ir.create_func
    (List_type [t; t'])
    (fun [x; y] ->
      let f, _, _ = f x in
      f y )
