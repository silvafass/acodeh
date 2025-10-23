use std::{
    collections::VecDeque,
    fs::ReadDir,
    io::Result,
    path::{Path, PathBuf},
};

#[derive(Debug)]
enum InnerEntryPath {
    Path(PathBuf),
    DeferredPath(PathBuf),
}

#[derive(Default, Debug)]
struct FileSearcherOptions {
    overall: bool,
    max_depth: usize,
    includes: Vec<String>,
    excludes: Vec<String>,
    extensions: Vec<String>,
}

#[derive(Debug, Default)]
pub struct FileSearcher {
    start_path: PathBuf,
    options: FileSearcherOptions,
}

impl FileSearcher {
    pub fn new<P: AsRef<Path>>(start_path: P) -> Self {
        let start_path = start_path.as_ref().to_path_buf();
        if start_path.is_file() || start_path.is_dir() {
            FileSearcher {
                start_path,
                options: FileSearcherOptions {
                    max_depth: usize::MAX,
                    ..FileSearcherOptions::default()
                },
            }
        } else {
            FileSearcher {
                options: FileSearcherOptions {
                    max_depth: usize::MAX,
                    ..FileSearcherOptions::default()
                },
                ..FileSearcher::default()
            }
        }
    }

    pub fn overall(mut self, flag: bool) -> Self {
        self.options.overall = flag;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.options.max_depth = max_depth;
        self
    }

    pub fn includes<P: AsRef<Path>>(mut self, includes: &[P]) -> Self {
        self.options.includes = includes
            .iter()
            .map(|item| item.as_ref().to_path_buf().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        self
    }

    pub fn excludes<P: AsRef<Path>>(mut self, excludes: &[P]) -> Self {
        self.options.excludes = excludes
            .iter()
            .map(|item| item.as_ref().to_path_buf().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        self
    }

    pub fn extensions(mut self, extensions: Option<impl AsRef<str>>) -> Self {
        self.options.extensions = extensions
            .map(|value| {
                value
                    .as_ref()
                    .split(&[',', ';', '|', ' '][..])
                    .map(|item| item.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        self
    }
}

impl IntoIterator for FileSearcher {
    type Item = Result<PathBuf>;

    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            options: self.options,
            offset_depth: self.start_path.components().count(),
            pending_paths: VecDeque::from([InnerEntryPath::Path(self.start_path)]),
            current_read_directory: None,
        }
    }
}

#[derive(Debug)]
pub struct IntoIter {
    options: FileSearcherOptions,
    pending_paths: VecDeque<InnerEntryPath>,
    current_read_directory: Option<ReadDir>,
    offset_depth: usize,
}

impl IntoIter {
    fn inner_next(&mut self) -> Option<Result<PathBuf>> {
        while !self.pending_paths.is_empty() || self.current_read_directory.is_some() {
            if let Some(read_dir) = &mut self.current_read_directory {
                for entry_result in read_dir {
                    match entry_result {
                        Ok(entry) => {
                            let path = entry.path();
                            let current_depth = path.components().count() - self.offset_depth;
                            if (path.is_file() || path.is_dir())
                                && current_depth <= self.options.max_depth
                            {
                                self.pending_paths.push_front(InnerEntryPath::Path(path));
                            }
                        }
                        Err(error) => return Some(Err(error)),
                    }
                }
                self.current_read_directory = None;
            } else if let Some(entry_path) = self.pending_paths.pop_front() {
                match entry_path {
                    InnerEntryPath::DeferredPath(pending_path) => return Some(Ok(pending_path)),
                    InnerEntryPath::Path(pending_path) => {
                        if pending_path.is_dir() {
                            match pending_path.read_dir() {
                                Ok(read_dir) => {
                                    self.current_read_directory = Some(read_dir);
                                }
                                Err(error) => return Some(Err(error)),
                            }
                            if self.options.overall {
                                self.pending_paths
                                    .push_front(InnerEntryPath::DeferredPath(pending_path));
                            } else {
                                return Some(Ok(pending_path));
                            }
                        } else {
                            return Some(Ok(pending_path));
                        }
                    }
                }
            } else {
                return None;
            }
        }
        None
    }
}

impl Iterator for IntoIter {
    type Item = Result<PathBuf>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(result) = self.inner_next() {
            let path = match result {
                Ok(path) => path,
                Err(error) => return Some(Err(error)),
            };

            let to_excludes = if self.options.excludes.is_empty() {
                false
            } else {
                self.options
                    .excludes
                    .iter()
                    .any(|item| path.to_string_lossy().contains(&item[..]))
            };
            if to_excludes {
                if path.is_dir() {
                    self.skip_current_directory();
                }
                continue;
            }

            let to_includes = if self.options.includes.is_empty() {
                true
            } else {
                self.options
                    .includes
                    .iter()
                    .any(|item| path.to_string_lossy().contains(&item[..]))
            };
            if !to_includes {
                continue;
            }

            let to_includes_extensions = if self.options.extensions.is_empty() {
                true
            } else if let Some(file_extension) = path.extension() {
                self.options
                    .extensions
                    .iter()
                    .any(|item| &file_extension.to_string_lossy() == item)
            } else {
                false
            };
            if !to_includes_extensions {
                continue;
            }

            return Some(Ok(path));
        }
        None
    }
}

impl IntoIter {
    pub fn filter_path<P: FnMut(&PathBuf) -> bool>(self, predicate: P) -> FilterPath<Self, P> {
        FilterPath {
            inner: self,
            predicate,
        }
    }

    pub fn skip_current_directory(&mut self) {
        self.current_read_directory = None;
    }
}

#[derive(Debug)]
pub struct FilterPath<I, P> {
    inner: I,
    predicate: P,
}

impl<P> Iterator for FilterPath<IntoIter, P>
where
    P: FnMut(&PathBuf) -> bool,
{
    type Item = Result<PathBuf>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(result) = self.inner.next() {
            let path = match result {
                Ok(path) => path,
                Err(error) => return Some(Err(error)),
            };

            if !(self.predicate)(&path) {
                if path.is_dir() {
                    self.inner.skip_current_directory();
                }
                continue;
            }

            return Some(Ok(path));
        }
        None
    }
}

impl<P> FilterPath<IntoIter, P>
where
    P: FnMut(&PathBuf) -> bool,
{
    pub fn filter_path(self, predicate: P) -> FilterPath<Self, P> {
        FilterPath {
            inner: self,
            predicate,
        }
    }

    pub fn skip_current_directory(&mut self) {
        self.inner.current_read_directory = None;
    }
}
