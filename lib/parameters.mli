type ('a, 'b, 'c) t

val return : 'a Ir.Var.t -> ('b, 'b, 'a) t

val bind : ('a, 'b, 'c) t -> ('c Ir.Var.t -> ('b, 'd, 'e) t) -> ('a, 'd, 'e) t

val ( let* ) :
  ('a, 'b, 'c) t -> ('c Ir.Var.t -> ('b, 'd, 'e) t) -> ('a, 'd, 'e) t

val new_param :
  ('a, Runtime.Value.host) Runtime.Value.t -> ('b, 'a Ir.Var.t -> 'b, 'a) t

val apply : (unit, 'a, 'b) t -> 'a Ir.VarList.t Ir.Var.t -> 'b Ir.Var.t

val param_type :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> (unit, 'b, 'c) t)
  -> 'b Ir.VarList.t Ir.ValueType.t

val grad_and_value :
     ('a, 'b, 'c) t
  -> ('a, 'b, ('b Ir.VarList.t Ir.Var.t -> 'c Ir.Var.t -> unit) Ir.VarList.t) t

val params : ('a, 'a, 'a Ir.VarList.t) t

val create_func :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> (unit, 'b, 'c) t)
  -> ( (   'b Ir.VarList.t Ir.Var.t
        -> 'a Ir.Var.t
        -> (Ir.u64, Unsigned.uint64) Ir.tensor Ir.Var.t
        -> unit )
       Ir.VarList.t
     , ('c Ir.Var.t -> (Ir.u64, Unsigned.uint64) Ir.tensor Ir.Var.t -> unit)
       Ir.VarList.t )
     Ir.Func.t

val initial :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> (unit, 'b, 'c) t)
  -> ('b Ir.VarList.t, Runtime.Value.host) Runtime.Value.t
