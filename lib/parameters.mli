type ('a, 'b) t

val return : 'a Ir.Var.t -> (unit Hlist.hlist, 'a) t

val bind :
     ('a, 'b) t
  -> ('b Ir.Var.t -> ('c Hlist.hlist, 'd) t)
  -> (('a -> 'c) Hlist.hlist, 'd) t

val ( let* ) :
     ('a, 'b) t
  -> ('b Ir.Var.t -> ('c Hlist.hlist, 'd) t)
  -> (('a -> 'c) Hlist.hlist, 'd) t

val new_param : 'a Runtime.HostValue.t -> ('a, 'a) t

val to_fun : ('a, 'b) t -> 'a Ir.Var.t -> 'b Ir.Var.t

val initial :
  'a Ir.ValueType.t -> ('a Ir.Var.t -> ('b, 'c) t) -> 'b Runtime.HostValue.t

val params_for : ('a, 'b) t -> ('a, 'a) t

val param_type :
  'a Ir.ValueType.t -> ('a Ir.Var.t -> ('b, 'c) t) -> 'b Ir.ValueType.t
