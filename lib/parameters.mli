type ('a, 'b) t

val return : 'a -> (unit Hlist.hlist, 'a) t

val bind :
  ('a, 'b) t -> ('b -> ('c Hlist.hlist, 'd) t) -> (('a * 'c) Hlist.hlist, 'd) t

val ( let* ) :
  ('a, 'b) t -> ('b -> ('c Hlist.hlist, 'd) t) -> (('a * 'c) Hlist.hlist, 'd) t

val new_param : 'a Runtime.HostValue.t -> ('a, 'a Ir.Var.t) t

val to_fun : ('a, 'b) t -> 'a Ir.Var.t -> 'b

val initial :
  'a Ir.ValueType.t -> ('a Ir.Var.t -> ('b, 'c) t) -> 'b Runtime.HostValue.t

val params_for : ('a, 'b) t -> ('a, 'a Ir.Var.t) t

val param_type :
  'a Ir.ValueType.t -> ('a Ir.Var.t -> ('b, 'c) t) -> 'b Ir.ValueType.t

val flatten : (('a * unit) Hlist.hlist, 'b) t -> ('a, 'b) t
