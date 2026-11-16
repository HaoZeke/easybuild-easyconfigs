import ast, hashlib, json, os, pathlib, traceback
import torch
from torch._subclasses.fake_tensor import DynamicOutputShapeException
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import aot_autograd_check as original_check
from unittest.mock import patch

root=pathlib.Path(__file__).resolve().parent.parent
out=pathlib.Path(__file__).resolve().parent
src=root/"v2.10.0/test/functorch/test_aotdispatch.py"
raw=src.read_bytes()
assert hashlib.sha256(raw).hexdigest()=="c8edff4474b2ba81850dd71702c88b8eb1e3d61e1c21a462d965617bfa39af39"
tree=ast.parse(raw)
helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_test_aot_autograd_helper")
op=next(o for o in op_db if o.name=="linalg.pinv" and o.variant_test_name=="hermitian")
events=[]
seed=int(os.environ.get("PINV_SEED","1234"))
torch.manual_seed(seed)
torch.set_num_threads(4)
def observed_check(func,args,kwargs,*rest,**kw):
    index=len(events)
    inp=args[0]
    item={"sample":index,"shape":list(inp.shape),"stride":list(inp.stride()),"kwargs":repr(kwargs),"dtype":str(inp.dtype),"grad":inp.requires_grad}
    try:
        result=original_check(func,args,kwargs,*rest,**kw)
    except BaseException as error:
        item.update({"status":"FAIL","exception":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        torch.save({"args":args,"kwargs":kwargs,"rng":torch.get_rng_state()},out/f"failed-sample-{index}-seed-{seed}.pt")
        events.append(item)
        raise
    item["status"]="PASS";events.append(item)
    print(json.dumps(item),flush=True)
    return result
namespace={"torch":torch,"DynamicOutputShapeException":DynamicOutputShapeException,"GuardOnDataDependentSymNode":GuardOnDataDependentSymNode,"aot_autograd_check":observed_check}
exec(compile(ast.Module(body=[helper],type_ignores=[]),str(src),"exec"),namespace)
case=TestCase("runTest")
case.setUp()
status=0
try:
    with patch("functorch.compile.config.debug_assert",True):
        namespace["_test_aot_autograd_helper"](case,"cpu",torch.float32,op,dynamic=True)
except BaseException:
    status=1
finally:
    case.tearDown()
result={"torch":torch.__version__,"seed":seed,"threads":torch.get_num_threads(),"source_sha256":hashlib.sha256(raw).hexdigest(),"unchanged_helper_ast_sha256":hashlib.sha256(ast.dump(helper,include_attributes=False).encode()).hexdigest(),"status":status,"samples":events,"backend_mkl":torch.backends.mkl.is_available(),"scope":"Released helper and OpInfo sampling against public CPU wheel; native EESSI failing sample not established."}
dest=out/f"result-seed-{seed}.json"
dest.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"result":str(dest),"status":status,"samples":len(events),"failures":sum(x["status"]!="PASS" for x in events)}),flush=True)
raise SystemExit(status)
