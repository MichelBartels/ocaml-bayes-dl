type ('a, 'b, 'c) t

val return : 'a Ir.Var.t -> ('b, 'b, 'a) t

val bind : ('a, 'b, 'c) t -> ('c Ir.Var.t -> ('b, 'd, 'e) t) -> ('a, 'd, 'e) t

val ( let* ) :
  ('a, 'b, 'c) t -> ('c Ir.Var.t -> ('b, 'd, 'e) t) -> ('a, 'd, 'e) t

val new_param : 'a Runtime.HostValue.t -> ('b, 'a -> 'b, 'a) t

val apply : (unit, 'a, 'b) t -> 'a Hlist.hlist Ir.Var.t -> 'b Ir.Var.t

val param_type :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> (unit, 'b, 'c) t)
  -> 'b Hlist.hlist Ir.ValueType.t

val grad_and_value :
  ('a, 'b, 'c) t -> ('a, 'b, ('b Hlist.hlist -> 'c -> unit) Hlist.hlist) t

val params : ('a, 'a, 'a Hlist.hlist) t

val create_func :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> (unit, 'b, 'c) t)
  -> ( (   'b Hlist.hlist
        -> 'a
        -> (Ir.Tensor.u64, Unsigned.uint64) Hlist.element
        -> unit )
       Hlist.hlist
     , ('c -> (Ir.Tensor.u64, Unsigned.uint64) Hlist.element -> unit)
       Hlist.hlist )
     Ir.Func.t

val initial :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> (unit, 'b, 'c) t)
  -> 'b Hlist.hlist Runtime.HostValue.t
