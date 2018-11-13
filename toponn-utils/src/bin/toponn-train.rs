extern crate conllx;

extern crate failure;

extern crate getopts;

extern crate stdinout;

extern crate toponn;

extern crate toponn_utils;

use std::env::args;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process;

use conllx::ReadSentence;
use failure::Error;
use getopts::Options;
use stdinout::OrExit;

use toponn::{Collector, HDF5Collector, Numberer, SentVectorizer};
use toponn_utils::{CborRead, CborWrite, Config, TomlRead};

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] CONFIG DATA OUTPUT.HDF5", program);
    print!("{}", opts.usage(&brief));
    process::exit(1);
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse options", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.len() != 3 {
        print_usage(&program, opts);
        return;
    }

    let config_file = File::open(&matches.free[0]).or_exit(
        format!("Cannot open configuration file '{}'", &matches.free[0]),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(&matches.free[0])
        .or_exit("Cannot relativize paths in configuration", 1);

    let input_file = File::open(&matches.free[1])
        .or_exit(format!("Cannot open '{}' for reading", &matches.free[0]), 1);
    let reader = conllx::Reader::new(BufReader::new(input_file));

    let labels = load_labels_or_new(&config).or_exit(
        format!(
            "Cannot load or create label file '{}'",
            config.labeler.labels
        ),
        1,
    );

    let embeddings = config
        .embeddings
        .load_embeddings()
        .or_exit("Cannot load embeddings", 1);
    let vectorizer = SentVectorizer::new(embeddings);

    let mut collector = HDF5Collector::new(
        &matches.free[2],
        labels,
        vectorizer,
        config.model.batch_size,
    ).or_exit(format!("Cannot create HDF file '{}'", &matches.free[2]), 1);

    for sentence in reader.sentences() {
        let sentence = sentence.or_exit("Cannot parse sentence", 1);
        collector
            .collect(&sentence)
            .or_exit("Cannot collect sentence", 1);
    }

    write_labels(&config, collector.labels()).or_exit("Cannot write labels", 1);
}

fn load_labels_or_new(config: &Config) -> Result<Numberer<String>, Error> {
    let labels_path = Path::new(&config.labeler.labels);
    if !labels_path.exists() {
        return Ok(Numberer::new(1));
    }

    eprintln!("Loading labels from: {:?}", labels_path);

    let f = File::open(labels_path)?;
    let system = Numberer::from_cbor_read(f)?;

    Ok(system)
}

fn write_labels(config: &Config, labels: &Numberer<String>) -> Result<(), Error> {
    let labels_path = Path::new(&config.labeler.labels);
    if labels_path.exists() {
        return Ok(());
    }

    let mut f = File::create(labels_path)?;
    labels.to_cbor_write(&mut f)?;
    Ok(())
}
