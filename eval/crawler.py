import os


class ResultsCrawler:
    """
    A crawler for the file structures in results folders, usually adhering to the pattern
    <data stream>/<detector/experiment>.csv.
    """
    def __init__(
        self,
        base_dir: str,
    ):
        """
        Init a new ResultsCrawler.

        :param base_dir: the base directory containing the results to be crawled
        """
        self.base_dir = base_dir

    def crawl(self, sub_path: str = ""):
        """
        Crawl the results directory and return the paths to all results files encountered in the directory.

        :param sub_path: the starting point for a new crawl for recursive use
        :return: the full path to results files
        """
        full_path = os.path.join(self.base_dir, sub_path)
        files = sorted(os.listdir(full_path))
        for file_ in files:
            file_path = os.path.join(full_path, file_)
            if os.path.isdir(file_path):
                for sub_file in self.crawl(os.path.join(sub_path, file_)):
                    yield sub_file
            elif os.path.splitext(file_path)[1] == ".csv":
                yield os.path.join(sub_path, file_)


if __name__ == "__main__":
    crawler = ResultsCrawler("results")
    for f in crawler.crawl():
        print(f)
