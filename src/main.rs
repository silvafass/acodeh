use acodeh::{fs::FileSearcher, llm};
use clap::Parser;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

const DEFAULT_MAX_CONTEXT: u64 = 16 * 1_024;

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
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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

            let mut total_content_len = 0;
            let mut documents = vec![];
            for path in paths_iter {
                let extension = path
                    .extension()
                    .map(|extension| extension.to_string_lossy().to_string())
                    .unwrap_or_default();
                let path_as_string = path.to_string_lossy().to_string();

                let content = if path.extension().unwrap_or_default().to_string_lossy() == "pdf" {
                    pdf_extract::extract_text(&path)
                        .inspect_err(|error| {
                            if debug {
                                eprintln!("{error:?}\nwhile reading {path_as_string}");
                            }
                        })
                        .ok()
                } else {
                    tokio::fs::read_to_string(&path)
                        .await
                        .inspect_err(|error| {
                            if debug {
                                eprintln!("{error:?}\nwhile reading {path_as_string}");
                            }
                        })
                        .ok()
                };

                if let Some(content) = content {
                    let content = format!(
                        "path: {}\n```{}\n{}\n```",
                        path_as_string, extension, content
                    );

                    if let Some(max_context) = max_context.or(Some(DEFAULT_MAX_CONTEXT))
                        && (total_content_len + content.len() as u64) / 4 > max_context
                    {
                        break;
                    }
                    total_content_len += content.len() as u64;

                    documents.push((path, content));
                }
            }

            let document_count = documents.len();
            let context = documents
                .into_iter()
                .inspect(|(path, ..)| {
                    if debug {
                        println!("Adding file {path:?}");
                    }
                })
                .map(|(.., content)| content)
                .reduce(|acc, content| format!("{acc}\n{content}"))
                .unwrap_or_default();
            let prompt = format!(include_str!("prompt_templete.in"), prompt, context);

            let context_len_estimated = total_content_len / 4;
            let prompt_context_len_estimated: u64 = prompt.len() as u64 / 4;

            let max_context = match max_context {
                Some(max_context) => max_context,
                _ if prompt_context_len_estimated > DEFAULT_MAX_CONTEXT => DEFAULT_MAX_CONTEXT,
                _ => {
                    let mut aligned_context_len = 2 * 1024;
                    while aligned_context_len < prompt_context_len_estimated {
                        aligned_context_len *= 2;
                    }

                    if aligned_context_len > DEFAULT_MAX_CONTEXT {
                        DEFAULT_MAX_CONTEXT
                    } else {
                        aligned_context_len
                    }
                }
            };

            println!("\n{:#^80}", " Payload stats ");
            println!("Total of documents {document_count}");
            println!("Total of documents contents: {total_content_len}");
            println!("Documents context size: {context_len_estimated}");
            println!("Prompt context size: {prompt_context_len_estimated}");
            println!("Max Context size: {max_context}");
            println!("{:#^80}\n", "");

            let llm = llm::LLMClient::default();

            let is_stream = true;

            let payload = llm::GeneratePayload {
                model: model.unwrap_or("llama3.2:latest".to_string()),
                system: Some(include_str!("system.in").to_string()),
                prompt: Some(prompt),
                stream: Some(is_stream),
                options: Some(llm::ModelParameters {
                    num_ctx: Some(max_context),
                }),
            };

            if is_stream {
                llm.generate_stream(&payload, |chunk| async move {
                    print!("{}", chunk.response);
                    std::io::stdout().flush().unwrap();
                    if chunk.done {
                        println!("\n\n{:#^80}", " Reponse stats ");
                        println!("model: {}", chunk.model);
                        println!("eval_count: {}", chunk.eval_count);
                        println!("prompt_eval_count: {}", chunk.prompt_eval_count);
                        println!("error: {:?}", chunk.error);
                        println!(
                            "total_duration: {:?}",
                            Duration::from_nanos(chunk.total_duration)
                        );
                        println!("{:#^80}\n", "");

                        if debug {
                            println!("\n{:#?}", chunk);
                        }
                    }
                    chunk.done
                })
                .await?;
            } else {
                let generated = llm.generate_once(&payload).await?;
                println!("{}", generated.response);

                println!("\n\n{:#^80}", " Reponse stats ");
                println!("model: {}", generated.model);
                println!("eval_count: {}", generated.eval_count);
                println!("prompt_eval_count: {}", generated.prompt_eval_count);
                println!("error: {:?}", generated.error);
                println!(
                    "total_duration: {:?}",
                    Duration::from_nanos(generated.total_duration)
                );
                println!("{:#^80}\n", "");

                if debug {
                    println!("\n{:#?}", generated);
                }
            }
        }
    }

    Ok(())
}
