extern crate conllx;
extern crate getopts;
extern crate stdinout;
extern crate tensorflow;
extern crate threadpool;
extern crate toponn;
extern crate toponn_utils;

use std::cell::RefCell;
use std::env::args;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process;
use std::sync::Arc;

use conllx::ReadSentence;
use getopts::Options;
use stdinout::OrExit;
use tensorflow::Session;
use threadpool::ThreadPool;

use toponn::{Numberer, SentVectorizer};
use toponn::tensorflow::{Model, Tagger};
use toponn_utils::{CborRead, Config, Result, SentProcessor, TomlRead};

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] CONFIG ADDR", program);
    print!("{}", opts.usage(&brief));
    process::exit(1);
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    opts.optopt(
        "t",
        "threads",
        "default server threadpool size (default: 4)",
        "N",
    );
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse options", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.len() != 2 {
        print_usage(&program, opts);
        return;
    }

    let n_threads = matches
        .opt_str("t")
        .as_ref()
        .map(|t| {
            t.parse()
                .or_exit(format!("Invalid number of threads: {}", t), 1)
        })
        .unwrap_or(4);

    let config_file = File::open(&matches.free[0]).or_exit(
        format!("Cannot open configuration file '{}'", &matches.free[0]),
        1,
    );

    let addr = &matches.free[1];

    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(&matches.free[0])
        .or_exit("Cannot relativize paths in configuration", 1);

    let labels = load_labels(&config).or_exit(
        format!("Cannot load label file '{}'", config.labeler.labels),
        1,
    );

    let embeddings = config
        .embeddings
        .load_embeddings()
        .or_exit("Cannot load embeddings", 1);
    let vectorizer = SentVectorizer::new(embeddings);

    let graph_reader = File::open(&config.model.filename).or_exit(
        format!(
            "Cannot open computation graph '{}' for reading",
            &config.model.filename
        ),
        1,
    );

    let model = Arc::new(
        Model::load_graph(graph_reader, vectorizer, labels, &config.model)
            .or_exit("Cannot load computation graph", 1),
    );

    let pool = ThreadPool::new(n_threads);

    let listener = TcpListener::bind(addr).or_exit(format!("Cannot listen on '{}'", addr), 1);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let config = config.clone();
                let model = model.clone();
                pool.execute(move || handle_client(config, model, stream))
            }
            Err(err) => eprintln!("Error processing stream: {}", err),
        }
    }
}

thread_local!{
    static SESSION: RefCell<Option<Session>> = RefCell::new(None);
}

fn handle_client(config: Config, model: Arc<Model>, stream: TcpStream) {
    let conllx_stream = match stream.try_clone() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("Cannot clone stream: {}", err);
            return;
        }
    };

    SESSION.with(|cell| {
        let mut session_ref = cell.borrow_mut();
        let mut session = session_ref.get_or_insert(
            config
                .model
                .new_session(&model)
                .or_exit("Cannot create Tensorflow session", 1),
        );

        let mut tagger = Tagger::new(&config.model, &model, &mut session);

        let reader = conllx::Reader::new(BufReader::new(&conllx_stream));
        let writer = conllx::Writer::new(BufWriter::new(&conllx_stream));

        let sent_proc = SentProcessor::new(&mut tagger, writer, config.model.batch_size);

        process_sentences(reader, sent_proc, stream);
    });
}

fn process_sentences<R, W>(
    reader: conllx::Reader<R>,
    mut sent_proc: SentProcessor<W>,
    mut stream: TcpStream,
) where
    R: BufRead,
    W: Write,
{
    for sentence in reader.sentences() {
        let sentence = match sentence {
            Ok(sentence) => sentence,
            Err(err) => {
                if let Err(err) = writeln!(stream, "Error writing sentence: {}", err) {
                    eprintln!("Error writing to client: {}", err);
                }

                sent_proc.clear();

                break;
            }
        };

        if let Err(err) = sent_proc.process(sentence) {
            if let Err(err) = writeln!(stream, "Error processing sentence: {}", err) {
                eprintln!("Error writing to client: {}", err);
            }

            sent_proc.clear();

            break;
        }
    }
}

fn load_labels(config: &Config) -> Result<Numberer<String>> {
    let labels_path = Path::new(&config.labeler.labels);

    eprintln!("Loading labels from: {:?}", labels_path);

    let f = File::open(labels_path)?;
    Numberer::from_cbor_read(f)
}
