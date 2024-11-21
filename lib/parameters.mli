type ('a, 'b, 'c, 'd, 'e) t

val return :
  'a Ir.Var.t -> (unit, 'a, 'b, unit Ir.VarList.t Ir.Var.t -> 'b, 'c) t

val bind :
     ('f, 'g, 'h, 'c Ir.VarList.t Ir.Var.t -> 'h, 'e) t
  -> ('g Ir.Var.t -> ('a, 'b, 'c, 'd, 'e) t)
  -> ( 'a Ir.VarList.t Ir.Var.t -> 'f
     , 'b
     , 'h
     , 'd Ir.VarList.t Ir.Var.t -> 'h
     , 'e )
     t

val ( let* ) :
     ('f, 'g, 'h, 'c Ir.VarList.t Ir.Var.t -> 'h, 'e) t
  -> ('g Ir.Var.t -> ('a, 'b, 'c, 'd, 'e) t)
  -> ( 'a Ir.VarList.t Ir.Var.t -> 'f
     , 'b
     , 'h
     , 'd Ir.VarList.t Ir.Var.t -> 'h
     , 'e )
     t

val new_param :
     'a Ir.tensor Ir.ValueType.t
  -> ( 'a Ir.tensor Ir.Var.t -> unit
     , 'a Ir.tensor
     , 'b
     , ('c Ir.Var.t -> unit) Ir.VarList.t Ir.Var.t -> 'b
     , 'c )
     t

val grad_and_value :
     ('a, 'b, 'b Ir.Var.t -> unit, 'c, 'b) t
  -> ( 'a
     , ('a Ir.VarList.t Ir.Var.t -> 'c) Ir.VarList.t
     , 'b Ir.Var.t -> unit
     , 'c
     , 'b )
     t

val create_func :
     'a Ir.ValueType.t
  -> ('a Ir.Var.t -> ('b, 'c, 'd, 'e, 'f) t)
  -> ( ('a Ir.Var.t -> 'b Ir.VarList.t Ir.Var.t -> unit) Ir.VarList.t
     , 'c )
     Ir.Func.t
