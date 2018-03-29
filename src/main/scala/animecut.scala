import org.opencv.core._
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import org.opencv.imgproc.Imgproc
import org.opencv.imgcodecs.Imgcodecs
import collection.JavaConverters._
import collection.mutable._

object Main{
  def main(args: Array[String]){
    val filename = "/home/satoru/storage/movie/ryuoh-anime/op.mp4"
    val out_dif  = "/home/satoru/storage/movie/ryuoh-anime/out"
    val start = 30
    val end = 60
    proc(filename, out_dif, start, end)
  }
  def proc(filename: String, out_dif: String, start: Int, end: Int){
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val cap = new VideoCapture(filename)
    val fps: Int = cap.get(Videoio.CAP_PROP_FPS).toInt
    var cut_points:List[Int] = List()
    
    var pre_frame = new Mat()
    cap.read(pre_frame)
    Imgproc.resize(pre_frame, pre_frame, new Size(256, 144))
    Imgproc.cvtColor(pre_frame, pre_frame, Imgproc.COLOR_RGB2GRAY)
    var counter = end - start
    while(cap.isOpened() && counter >= 0) {
      var post_frame = new Mat()
      cap.read(post_frame)
      Imgproc.resize(post_frame, post_frame, new Size(256, 144))
      Imgproc.cvtColor(post_frame, post_frame, Imgproc.COLOR_RGB2GRAY)
      if(iscut(pre_frame, post_frame)){
        cut_points = cut_points :+ cap.get(Videoio.CAP_PROP_POS_FRAMES).toInt
      }
      counter -= 1
      pre_frame = post_frame.clone()
      post_frame.release()
    }
    println(cut_points.mkString(" "))
  }
  def iscut(pre: Mat, post: Mat):Boolean={
    if(iscut_absdiff(pre, post)){
      return true
    }
    if(iscut_blockdiff(pre, post)){
      return true
    }
    return false
  }
  def iscut_absdiff(pre: Mat, post: Mat):Boolean={
    var diff = new Mat()
    Core.absdiff(pre, post, diff)
    val diffval:Scalar = Core.mean(diff)
    diff.release
    if(diffval.`val`(0) > 40){
      return true
    } else {
      return false
    }
  }
  def iscut_blockdiff(pre: Mat, post: Mat):Boolean={
    var count = 0
    for {i <- 1 to 16
         j <- 1 to 9}{
      val pre_roe = new Mat(pre, new Rect((i - 1) * 16, (j - 1) * 16, 16, 16))
      val pre_hist = get_hist(pre_roe)
      var diffs: Array[Double] = Array.empty 
      //postフレームのイテレーション
      for {k <- 1 to 16
           l <- 1 to 9}{
        val pos_roe = new Mat(post, new Rect((k - 1) * 16, (l - 1) * 16, 16, 16))
        val pos_hist = get_hist(pos_roe)
        val diff = new Mat()
        Core.absdiff(pos_hist, pre_hist, diff)
        val diffval = Core.mean(diff).`val`(0)
        diffs = diffs :+ diffval
      }
      if(diffs.min > 0.6){
        count += 1
      }
    }
    if(count > 30){
      return true
    }
    return false
  }
  def get_hist(src:Mat):org.opencv.core.Mat={
    val imgs: java.util.List[Mat] = ArrayBuffer(src).asJava
    var hist = new Mat()
    Imgproc.calcHist(
      imgs,
      new MatOfInt(0),
      new Mat(),
      hist,
      new MatOfInt(256),
      new MatOfFloat(0, 256)
    )
    return hist
  }
}