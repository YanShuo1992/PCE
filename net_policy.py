from __future__ import annotations

import os

BLOCKED_HOSTS = (
    "github.com", "www.github.com", "api.github.com",
    "raw.githubusercontent.com", "objects.githubusercontent.com",
    "codeload.github.com", "gist.github.com",
    "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "www.stackoverflow.com",
    "google.com", "www.google.com", "bing.com", "www.bing.com",
    "duckduckgo.com", "html.duckduckgo.com",
)

DEFERRED_HOSTS = (
    "pypi.org", "www.pypi.org", "pypi.python.org", "files.pythonhosted.org",
    "registry.npmjs.org", "registry.yarnpkg.com",
    "crates.io", "static.crates.io", "index.crates.io",
    "proxy.golang.org", "sum.golang.org", "goproxy.io",
    "repo.maven.apache.org", "repo1.maven.org", "jcenter.bintray.com",
)

ENV = "ALLOW_AGENT_NETWORK"

def enabled() -> bool:
    return (os.getenv(ENV, "") or "").strip().lower() not in ("1", "true", "yes", "on")

def extra_hosts() -> dict:
    return {h: "127.0.0.1" for h in BLOCKED_HOSTS} if enabled() else {}

def install() -> bool:
    if not enabled():
        return False
    from docker.models.containers import ContainerCollection
    if getattr(ContainerCollection, "_net_policy_installed", False):
        return True
    hosts = extra_hosts()
    for name in ("create", "run"):
        original = getattr(ContainerCollection, name)

        def wrapper(self, *a, _orig=original, **kw):
            if "extra_hosts" not in kw and kw.get("network_mode") != "none":
                kw["extra_hosts"] = dict(hosts)
            return _orig(self, *a, **kw)

        wrapper.__name__ = name
        wrapper.__doc__ = f"{original.__doc__ or ''}\n\n(net_policy: blocked hosts added)"
        setattr(ContainerCollection, name, wrapper)
    ContainerCollection._net_policy_installed = True
    return True

def seal(container, log=None) -> bool:
    if not enabled():
        return False
    say = log or (lambda _m: None)
    lines = "".join(f"127.0.0.1 {h}\\n::1 {h}\\n" for h in DEFERRED_HOSTS)
    res = container.exec_run(f"/bin/sh -c 'printf \"{lines}\" >> /etc/hosts'", user="root")
    if res.exit_code != 0:
        say(f"[net_policy] SEAL FAILED, /etc/hosts not written (exit {res.exit_code}): "
            f"{(res.output or b'').decode(errors='replace')[:200]}")
        return False

    def resolves(fam, host):
        r = container.exec_run(f"/bin/sh -c 'getent {fam} {host} || true'", user="root")
        return (r.output or b"").decode(errors="replace").strip()

    v4, v6 = resolves("ahostsv4", "pypi.org"), resolves("ahostsv6", "pypi.org")
    if not v4.startswith("127.0.0.1"):
        say(f"[net_policy] SEAL FAILED, pypi.org still resolves to {v4[:120] or '<nothing>'}")
        return False
    if v6 and not v6.startswith("::1"):
        say(f"[net_policy] SEAL FAILED over IPv6, pypi.org resolves to {v6[:120]}")
        return False
    say(f"[net_policy] sealed {len(DEFERRED_HOSTS)} hosts (v4+v6): "
        f"package indexes are closed for the agent")
    return True
