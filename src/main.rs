use acodeh::ollama::GenerateRequest;
use acodeh::{fs::FileSearcher, ollama, prompt::PromptBuilder};
use anyhow::anyhow;
use clap::Parser;
use futures::StreamExt;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Command {
    Run {
        prompt: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        path: Vec<PathBuf>,
        #[arg(long)]
        includes: Vec<PathBuf>,
        #[arg(long)]
        excludes: Vec<PathBuf>,
        #[arg(long)]
        extensions: Option<String>,
        #[arg(long)]
        overall: bool,
        #[arg(short, long)]
        recursive: bool,
        #[arg(long, default_value_t = 1)]
        max_depth: usize,
        #[arg(long)]
        max_context: Option<u64>,
        #[arg(long, default_value_t = false)]
        debug: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let command = Command::parse();

    match command {
        Command::Run {
            model,
            prompt,
            path,
            includes,
            excludes,
            extensions,
            overall,
            recursive,
            max_depth,
            max_context,
            debug,
        } => {
            let max_depth = if recursive { usize::MAX } else { max_depth };

            let paths_iter = path
                .iter()
                .flat_map(|start_path| {
                    let mut ignore_build = GitignoreBuilder::new(start_path);
                    if let Err(error) = ignore_build.add_line(None, ".git")
                        && debug
                    {
                        eprintln!("Could not add .git to ignore: {error:?}");
                    }

                    let mut ignore = match ignore_build.build() {
                        Ok(ignore) => ignore,
                        Err(error) => {
                            if debug {
                                eprintln!("Failed to build ignore patterns: {error:?}");
                                println!("Using a empty ignore pattern...");
                            }
                            Gitignore::empty()
                        }
                    };

                    FileSearcher::new(start_path)
                        .overall(overall)
                        .max_depth(max_depth)
                        .includes(&includes)
                        .excludes(&excludes)
                        .extensions(extensions.as_ref())
                        .into_iter()
                        .filter_path(move |path| {
                            if path.ends_with(".gitignore") {
                                if let Some(error) = ignore_build.add(path)
                                    && debug
                                {
                                    eprintln!("ERROR: {}", error);
                                }
                                ignore = match ignore_build.build() {
                                    Ok(ignore) => ignore,
                                    Err(error) => {
                                        if debug {
                                            eprintln!("Failed to build ignore patterns: {error:?}");
                                            println!("Using a empty ignore pattern...");
                                        }
                                        Gitignore::empty()
                                    }
                                };
                            }
                            ignore.matched(path, path.is_dir()).is_none()
                        })
                        .filter_map(|result| result.ok())
                })
                .filter(|path| path.is_file());

            let mut prompt_builder = PromptBuilder::new(prompt).max_context(max_context);
            for path in paths_iter {
                let path_as_string = path.to_string_lossy().to_string();
                let result = prompt_builder
                    .add_from_path(path)
                    .await
                    .inspect(|content_len| {
                        if debug {
                            println!("Adding file {path_as_string:?} ({}b)", content_len);
                        }
                    })
                    .inspect_err(|err| {
                        if debug {
                            eprintln!("{err:?}");
                        }
                    });
                if let Err(err) = result
                    && err.to_string().contains("Maximum context exceeded")
                {
                    break;
                }
            }

            let (prompt, prompt_stats) = prompt_builder.build()?;

            println!("\n{:#^80}", " Payload stats ");
            println!("\n{:#?}", prompt_stats);
            println!("{:#^80}\n", "");

            let client = ollama::LLMClient::default();

            let mut stream =
                GenerateRequest::new(&model.unwrap_or("llama3.2:latest".to_string()), &client)
                    .system(include_str!("system.in"))
                    .num_ctx_options(prompt_stats.max_context)
                    .prompt_stream(&prompt)
                    .await?;

            while let Some(response) = stream.next().await {
                if let Some(err) = response.error {
                    return Err(anyhow!("LLM error: {err}"));
                }

                print!("{}", response.response);
                std::io::stdout().flush().unwrap();
                if response.done {
                    println!("\n\n{:#^80}", " Reponse stats ");
                    println!("model: {}", response.model);
                    println!("eval_count: {}", response.eval_count);
                    println!("prompt_eval_count: {}", response.prompt_eval_count);
                    println!("error: {:?}", response.error);
                    println!(
                        "total_duration: {:?}",
                        Duration::from_nanos(response.total_duration)
                    );
                    println!("{:#^80}\n", "");

                    if debug {
                        println!("\n{:#?}", response);
                    }
                }
            }
        }
    }

    Ok(())
}
