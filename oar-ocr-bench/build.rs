use std::{env, fs, path::PathBuf, process::Command};

fn git(root: &std::path::Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("..");
    for path in ["HEAD", "index", "refs"] {
        if let Some(path) = git(&root, &["rev-parse", "--git-path", path]) {
            println!("cargo:rerun-if-changed={}", root.join(path).display());
        }
    }
    for path in [
        "src",
        "oar-ocr-core/src",
        "oar-ocr-vl/src",
        "oar-ocr-bench/src",
        "Cargo.lock",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    let commit = git(&root, &["rev-parse", "HEAD"]);
    let dirty = git(&root, &["status", "--porcelain"]).map(|status| !status.is_empty());
    let rustc = Command::new(env::var_os("RUSTC").unwrap())
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string());
    let generated = format!(
        "const BUILD_COMMIT: Option<&str> = {commit:?};\nconst BUILD_DIRTY: Option<bool> = {dirty:?};\nconst BUILD_RUSTC: Option<&str> = {rustc:?};\n"
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("build_info.rs"),
        generated,
    )
    .unwrap();
}
