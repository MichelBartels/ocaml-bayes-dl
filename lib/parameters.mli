type ('a, 'b) t

val return : 'a Ir.Var.t -> (unit, 'a) t

val bind :
     ('c, 'd) t
  -> ('d Ir.Var.t -> ('a, 'b) t)
  -> ('a Ir.VarList.t Ir.Var.t -> 'c, 'b) t

val ( let* ) :
     ('c, 'd) t
  -> ('d Ir.Var.t -> ('a, 'b) t)
  -> ('a Ir.VarList.t Ir.Var.t -> 'c, 'b) t

val new_param :
  'a Ir.tensor Ir.ValueType.t -> ('a Ir.tensor Ir.Var.t -> unit, 'a Ir.tensor) t

val grad_and_value :
     ('a, 'b) t
  -> ( 'a
     , (   'a Ir.VarList.t Ir.Var.t
        -> 'a Ir.VarList.t Ir.Var.t
        -> 'b Ir.Var.t
        -> unit )
       Ir.VarList.t )
     t

val create_func :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> ('b, 'c) t)
  -> ( ('a Ir.Var.t -> 'b Ir.VarList.t Ir.Var.t -> unit) Ir.VarList.t
     , 'c )
     Ir.Func.t
