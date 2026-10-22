"""以最小依赖导入 TerraTorch 的 TerraMind 官方实现。

为什么需要这个模块：
``import terratorch`` 会连带加载 datasets / datamodules / tasks / 全部 backbone，
把 albumentations、rasterio、geopandas、torchgeo、lightning 等整套地球观测栈都拉进来。
我们只需要 ``terramind`` 这一个 backbone，因此：

1. 优先走正常的 ``import terratorch...``（环境完整时行为完全一致）；
2. 若失败，则注册若干"路径占位"包，让 Python 直接按文件加载
   ``terratorch.models.backbones.terramind.*``，不执行其 ``__init__.py``；
3. 官方代码在 import 时会用 ``TERRATORCH_*_REGISTRY.register`` 做注册，
   而我们直接调用构建函数、不需要注册表，因此提供一个空注册表桩。

两种路径加载的都是**安装包内的同一份官方源码**，模型实现没有任何改动。
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


class _NullRegistry:
    """只支持 ``@REGISTRY.register`` 的空注册表。"""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, object] = {}

    def register(self, *args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            fn = args[0]
            self._items[getattr(fn, "__name__", str(fn))] = fn
            return fn

        def deco(fn):
            self._items[getattr(fn, "__name__", str(fn))] = fn
            return fn

        return deco

    def __contains__(self, key):
        return key in self._items

    def __getitem__(self, key):
        return self._items[key]

    def build(self, name, *a, **k):
        return self._items[name](*a, **k)


def _stub_registry() -> None:
    mod = types.ModuleType("terratorch.registry")
    names = (
        "BACKBONE_REGISTRY",
        "DECODER_REGISTRY",
        "MODEL_FACTORY_REGISTRY",
        "FULL_MODEL_REGISTRY",
        "TERRATORCH_BACKBONE_REGISTRY",
        "TERRATORCH_FULL_MODEL_REGISTRY",
        "TERRATORCH_DECODER_REGISTRY",
        "TERRATORCH_MODEL_FACTORY_REGISTRY",
    )
    for n in names:
        setattr(mod, n, _NullRegistry(n))
    sys.modules["terratorch.registry"] = mod


def _find_root() -> Path:
    import site

    for p in list(site.getsitepackages()) + [site.getusersitepackages()]:
        cand = Path(p) / "terratorch"
        if cand.is_dir():
            return cand
    raise ImportError("找不到 terratorch 安装目录")


def _stub_packages() -> None:
    try:
        import terratorch  # noqa: F401

        return  # 环境完整，无需 shim
    except Exception:
        pass

    root = _find_root()
    for name, path in (
        ("terratorch", root),
        ("terratorch.models", root / "models"),
        ("terratorch.models.backbones", root / "models" / "backbones"),
        ("terratorch.models.backbones.terramind", root / "models" / "backbones" / "terramind"),
    ):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod.__package__ = name
        sys.modules[name] = mod

    try:
        importlib.import_module("terratorch.registry")
    except Exception:
        _stub_registry()


def load_terramind():
    """返回 ``(TerraMindTiM, terramind_v1_small_tim, checkpoint_filter_fn_tim)``。"""
    _stub_packages()
    base = "terratorch.models.backbones.terramind.model"
    tim_mod = importlib.import_module(f"{base}.terramind_tim")
    reg = importlib.import_module(f"{base}.terramind_register")
    return tim_mod.TerraMindTiM, reg.terramind_v1_small_tim, reg.checkpoint_filter_fn_tim
