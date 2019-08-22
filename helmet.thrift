struct HelmetCheckResult {
  1: i32 errorCode
  2: i32 index
  3: string name
  4: double score 
}
service Helmet {
      HelmetCheckResult checkHelmet(1:string image)
}
