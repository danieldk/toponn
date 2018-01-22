extern crate conllx;
extern crate getopts;
extern crate stdinout;
extern crate tf_embed;
extern crate toponn;
extern crate toponn_utils;

use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::process;

use getopts::Options;
use stdinout::OrExit;
use tf_embed::{Embeddings, EmbeddingsBuilder, ReadWord2Vec, WriteWord2Vec};

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] INPUT OUTPUT", program);
    print!("{}", opts.usage(&brief));
    process::exit(1);
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse arguments", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.len() != 2 {
        print_usage(&program, opts);
        return;
    }

    let embed_file = File::open(&matches.free[0]).or_exit(
        format!("Cannot open embeddings '{}; for reading", &matches.free[0]),
        1,
    );
    let mut reader = BufReader::new(embed_file);
    let embeddings =
        Embeddings::read_word2vec_binary(&mut reader).or_exit("Cannot read word embeddings", 1);

    let null_embed_file =
        File::create(&matches.free[1]).or_exit("Cannot open embeddings for writing", 1);
    let mut writer = BufWriter::new(null_embed_file);

    let mut builder = EmbeddingsBuilder::new(embeddings.embed_len());
    builder.put(
        String::from("<NULL-TOKEN>"),
        vec![0f32; embeddings.embed_len()],
    );
    embeddings.iter().for_each(|(word, embedding)| {
        builder.put(word.to_owned(), embedding.to_owned());
    });

    let null_embeddings = builder.build();

    null_embeddings
        .write_word2vec_binary(&mut writer)
        .or_exit("Cannot write embeddings", 1);
}
