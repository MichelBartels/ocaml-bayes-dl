type ('a, 'b) t =
  ('a Ir.VarList.t Ir.Var.t -> 'b Ir.Var.t) * 'a Ir.VarList.t Ir.ValueType.t

let return : type a. a Ir.Var.t -> (unit, a) t =
 fun x -> ((fun _ -> x), List_type [])

let bind :
    type a b c d.
    (c, d) t -> (d Ir.Var.t -> (a, b) t) -> (a Ir.VarList.t Ir.Var.t -> c, b) t
    =
 fun (f, List_type t) g ->
  let _, t' = g (f (Ir.ValueType.to_arg (List_type t))) in
  ( (fun (x :: xs) ->
      let y = f xs in
      let z, _ = g y in
      z x )
  , List_type (t' :: t) )

let ( let* ) = bind

let new_param :
    type a.
    a Ir.tensor Ir.ValueType.t -> (a Ir.tensor Ir.Var.t -> unit, a Ir.tensor) t
    =
 fun t -> ((fun [x] -> x), List_type [t])

let grad_and_value :
    type a b.
       (a, b) t
    -> ( a
       , (   a Ir.VarList.t Ir.Var.t
          -> a Ir.VarList.t Ir.Var.t
          -> b Ir.Var.t
          -> unit )
         Ir.VarList.t )
       t =
 fun (f, t) -> ((fun p -> p :: Backpropagate.diff Var f p), t)

let create_func :
    type a b c.
       a Ir.ValueType.t
    -> (a Ir.Var.t -> (b, c) t)
    -> ( (a Ir.Var.t -> b Ir.VarList.t Ir.Var.t -> unit) Ir.VarList.t
       , c )
       Ir.Func.t =
 fun t f ->
  let args = Ir.ValueType.to_arg t in
  let _, t' = f args in
  Ir.create_func
    (List_type [t; t'])
    (fun [x; y] ->
      let f, _ = f x in
      f y )
