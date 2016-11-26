package io.github.benfradet

import java.io.FileWriter
import java.time.LocalDate
import java.time.format.DateTimeFormatter

import cats.syntax.either._
import com.twitter.util.{TimerTask, Duration, JavaTimer}
import io.circe.Decoder
import io.circe.generic.auto._
import io.circe.parser.decode
import io.circe.syntax._
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.io.Source

sealed trait SunSetRiseTrait {
  def sunrise: Option[String]
  def sunset: Option[String]
}
case class SunSetRise(sunrise: Option[String], sunset: Option[String]) extends SunSetRiseTrait
case class SunSetRiseDate(date: String, sunrise: Option[String], sunset: Option[String])
  extends SunSetRiseTrait
object SunSetRise {
  implicit val decodeSunSetRise: Decoder[SunSetRise] = Decoder.instance { c =>
    for {
      res <- c.downField("results").as[Map[String, String]]
    } yield SunSetRise(res.get("sunrise"), res.get("sunset"))
  }
}

/**
 * Build a dataset containing time of sunset and sunrise for the different dates contained in the
 * SF crime dataset.
 */
object BuildSetRiseDataset {
  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val sfLat = 37.7749d
    val sfLng = -122.4194d

    val minDate = LocalDate.of(2003, 1, 1)
    val maxDate = LocalDate.of(2015, 5, 14)
    val dateFormatter = DateTimeFormatter.ISO_LOCAL_DATE

    val urlTemplate = "http://api.sunrise-sunset.org/json?lat=%f&lng=%f&date=%s"
    val urlsQeueue = mutable.Queue(
      Util.daysBetween(minDate, maxDate)
        .map { d =>
          val formattedDate = d.format(dateFormatter)
          (formattedDate, urlTemplate.format(sfLat, sfLng, formattedDate))
        }: _*
    )
    require(urlsQeueue.nonEmpty, "there should be URLs to query")

    val fileWriter = new FileWriter("sfCrime/src/main/resources/sunsetrise.json")
    fileWriter.write("[\n")

    val timer = new JavaTimer(isDaemon = false)
    lazy val task: TimerTask = timer.schedule(Duration.fromSeconds(5)) {
      val (date, url) = urlsQeueue.dequeue()
      val sunSetRise = decode[SunSetRise](Source.fromURL(url).mkString)
        .map(s => SunSetRiseDate(date, s.sunrise, s.sunset))
      sunSetRise match {
        case Right(s) =>
          val json = s.asJson.spaces2
          if (urlsQeueue.isEmpty) {
            task.cancel()
            timer.stop()
            fileWriter.write(s"$json\n")
            fileWriter.write("]")
            fileWriter.close()
          } else {
            fileWriter.write(s"$json,\n")
          }
        case Left(e) => logger.warn("couldn't retrieve sunset/sunrise data", e)
      }
    }
    require(task != null)
  }
}
