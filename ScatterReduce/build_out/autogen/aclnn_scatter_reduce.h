
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_REDUCE_H_
#define ACLNN_SCATTER_REDUCE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterReduceGetWorkspaceSize
 * parameters :
 * self : required
 * index : required
 * src : required
 * dim : required
 * reduce : required
 * includeSelf : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterReduceGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *index,
    const aclTensor *src,
    int64_t dim,
    char *reduce,
    bool includeSelf,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterReduce
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterReduce(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
