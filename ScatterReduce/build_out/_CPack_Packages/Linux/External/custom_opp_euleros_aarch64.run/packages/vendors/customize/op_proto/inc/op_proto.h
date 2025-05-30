#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ScatterReduce)
    .INPUT(self, ge::TensorType::ALL())
    .INPUT(index, ge::TensorType::ALL())
    .INPUT(src, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(dim, Int)
    .REQUIRED_ATTR(reduce, String)
    .ATTR(include_self, Bool, true)
    .OP_END_FACTORY_REG(ScatterReduce);

}

#endif
