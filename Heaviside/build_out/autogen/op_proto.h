#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(Heaviside)
    .INPUT(input, ge::TensorType::ALL())
    .INPUT(values, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(Heaviside);

}

#endif
