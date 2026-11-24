import java.io.File
import java.io.PrintWriter
import scala.io.Source
import scala.util.Using
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import java.nio.file.StandardOpenOption.{APPEND, CREATE}
import scala.util.matching.Regex

def prettyPrintJson(jsonStr: String): String = {
  val sb = new StringBuilder
  var indent = 0
  var inString = false
  var escape = false
  jsonStr.foreach { c =>
    if (escape) { sb.append(c); escape = false }
    else c match {
      case '\\' => sb.append(c); escape = true
      case '"'  => sb.append(c); inString = !inString
      case '{' | '[' =>
        sb.append(c); if (!inString) { sb.append("\n"); indent += 2; sb.append(" " * indent) }
      case '}' | ']' =>
        if (!inString) { sb.append("\n"); indent -= 2; sb.append(" " * indent) }
        sb.append(c)
      case ',' =>
        sb.append(c); if (!inString) { sb.append("\n"); sb.append(" " * indent) }
      case ':' =>
        sb.append(c); if (!inString) sb.append(" ")
      case other => sb.append(other)
    }
  }
  sb.toString
}

def log_error(file_path: String, err_message: String): Unit = {
  val split_path = file_path.split('/')
  val out_path_base = split_path.slice(0, split_path.length - 3).mkString("/")
  val errPath = Paths.get(out_path_base, "log", "graph_export.log")
  try {
    Files.createDirectories(errPath.getParent)
    val line = s"$file_path\n    $err_message\n"
    Files.write(errPath, line.getBytes(StandardCharsets.UTF_8), CREATE, APPEND)
  } catch { case _: Throwable => () }
}

def listCFiles(dir: File): Seq[File] =
  Option(dir.listFiles).toSeq.flatten.filter(f => f.isFile && f.getName.toLowerCase.endsWith(".c"))

def listImmediateSubdirs(dir: File): Seq[File] =
  Option(dir.listFiles).toSeq.flatten.filter(_.isDirectory)

val EdgeRe: Regex = "\"(\\d+)\"\\s*->\\s*\"(\\d+)\"".r
def parseDotEdges(dotText: String): List[(Long, Long)] =
  EdgeRe.findAllMatchIn(dotText).map(m => (m.group(1).toLong, m.group(2).toLong)).toList

def pickDotFile(outDir: File, kind: String, defaultIndex: Int): Option[File] = {
  val files = Option(outDir.listFiles).toSeq.flatten
    .filter(f => f.isFile && f.getName.matches(s"\\d+-$kind\\.dot"))
    .sortBy(f => f.getName.split("-")(0).toInt)
  if (files.isEmpty) None
  else if (defaultIndex < files.length) Some(files(defaultIndex))
  else Some(files.last)
}

// ---------- Export one file: AST nodes and DOT-derived graph edges ----------
def export_one(file_path: String, output_dir: String): Unit = {
  // Read source text
  val src_string = Using.resource(Source.fromFile(file_path)) { src =>
    src.getLines().mkString("\n")
  }

  // Clear the working directory out/ before each export to remove stale DOT files
  val outDir = new File("out")
  if (outDir.exists()) {
    Option(outDir.listFiles).toSeq.flatten.foreach { f =>
      if (f.isDirectory) { Option(f.listFiles).toSeq.flatten.foreach(_.delete()); f.delete() }
      else f.delete()
    }
  } else outDir.mkdirs()

  // Build the CPG and export DOT graphs
  importCode.c.fromString(src_string)
  run.ossdataflow
  run.dumpast
  run.dumpcfg
  run.dumpcdg
  run.dumpddg

  // Use Joern AST JSON; node IDs are expected to match the DOT graphs
  val m = cpg.method.l
  if (m.isEmpty) throw new RuntimeException("No method parsed for: " + file_path)
  val meth = m(0)
  val ast_json = meth.ast.toJson

  // Read edges from the four DOT graph types
  def readDot(kind: String, idx: Int): List[(Long,Long)] = {
    pickDotFile(outDir, kind, idx) match {
      case None => Nil
      case Some(f) =>
        val text = Using.resource(Source.fromFile(f))(_.mkString)
        parseDotEdges(text)
    }
  }
  val ast_edges = readDot("ast", 0)   // Retain the historical AST DOT selection index
  val cfg_edges = readDot("cfg", 1)
  val cdg_edges = readDot("cdg", 1)
  val ddg_edges = readDot("ddg", 1)

  def toJsonPairs(es: List[(Long,Long)], reverse: Boolean = false): String = {
    val pairs = es.map { case (a,b) => if (reverse) s"[$b,$a]" else s"[$a,$b]" }
    pairs.mkString(",")
  }

  val out_str =
    s"""{
       |  "ast_nodes": $ast_json,
       |  "ast_edges": [${toJsonPairs(ast_edges, reverse = false)}],
       |  "cfg_edges": [${toJsonPairs(cfg_edges, reverse = false)}],
       |  "cdg_edges": [${toJsonPairs(cdg_edges, reverse = false)}],
       |  "ddg_edges": [${toJsonPairs(ddg_edges, reverse = false)}]
       |}""".stripMargin

  val pretty = prettyPrintJson(out_str)

  val file_name = new File(file_path).getName
  val out_file_name =
    if (file_name.toLowerCase.endsWith(".c")) file_name.dropRight(2) + ".json"
    else file_name + ".json"
  val outFile = new File(output_dir, out_file_name)
  outFile.getParentFile.mkdirs()
  new PrintWriter(outFile) { try { write(pretty) } finally { close() } }

  // Close the CPG and reset the workspace
  try { cpg.close() } catch { case _: Throwable => () }
  try { workspace.reset } catch { case _: Throwable => () } // reset takes no arguments
}

def export_wrapper(file_path: String, output_dir: String): Unit = {
  val file_name = new File(file_path).getName
  val out_file_name =
    if (file_name.toLowerCase.endsWith(".c")) file_name.dropRight(2) + ".json"
    else file_name + ".json"
  val outFile = new File(output_dir, out_file_name)

  if (outFile.exists() && outFile.length() > 0) {
    println(s"[skip] ${file_path} -> ${outFile.getAbsolutePath}")
  } else {
    try export_one(file_path, output_dir)
    catch { case e: Throwable => log_error(file_path, "joern error:\n        " + e) }
  }
}

def processBase(srcBase: String, outBase: String): Unit = {
  val srcRoot  = new File(srcBase)
  val outRoot  = new File(outBase)
  outRoot.mkdirs()

  // First process .c files directly under srcRoot
  val rootCFiles = listCFiles(srcRoot)
  println(s"[processBase] src=$srcBase  out=$outBase  rootCFiles=${rootCFiles.size}")

  if (rootCFiles.nonEmpty) {
    rootCFiles.foreach { f =>
      export_wrapper(f.getAbsolutePath, outRoot.getAbsolutePath)
    }
  }

  // Then process immediate subdirectories, if present
  val subdirs = listImmediateSubdirs(srcRoot)
  println(s"[processBase] subdirs=${subdirs.map(_.getName).mkString(",")}")

  subdirs.foreach { sub =>
    val subName = sub.getName
    val outDir  = new File(outRoot, subName)
    outDir.mkdirs()

    val cFiles = listCFiles(sub)
    println(s"  - ${subName}: ${cFiles.size} .c files")
    cFiles.foreach { f => export_wrapper(f.getAbsolutePath, outDir.getAbsolutePath) }
  }
}


@main def export_ast_ccd_from_dot(
  srcBaseArg: String = "",
  outBaseArg: String = ""
): Unit = {
  val defaultSrc = "./Location/"
  val defaultOut = "./Location_js/"

  val srcBase = Option(srcBaseArg).filter(_.nonEmpty)
    .orElse(sys.env.get("SRC_BASE"))
    .getOrElse(defaultSrc)

  val outBase = Option(outBaseArg).filter(_.nonEmpty)
    .orElse(sys.env.get("OUT_BASE"))
    .getOrElse(defaultOut)

  processBase(srcBase, outBase)
  println(s"[done] Exported to mirror under: $outBase")
}
