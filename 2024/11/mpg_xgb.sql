SELECT
  `mpg`,
  ((((((((((0.0 + CASE
WHEN ((`cyl` < 0.454947203 OR (`cyl` IS NULL))) THEN 6.67105246
WHEN (`cyl` >= 0.454947203) THEN 4.0880003
END) + CASE
WHEN ((`cyl` < -0.664922833 OR (`cyl` IS NULL))) THEN 5.36046076
WHEN ((`disp` < 1.7692467 OR (`disp` IS NULL)) AND `cyl` >= -0.664922833) THEN 3.35319948
WHEN (`disp` >= 1.7692467 AND `cyl` >= -0.664922833) THEN 1.16239989
END) + CASE
WHEN ((`wt` < -0.978325605 OR (`wt` IS NULL))) THEN 4.5090394
WHEN ((`disp` < 1.7692467 OR (`disp` IS NULL)) AND `wt` >= -0.978325605) THEN 2.52093935
WHEN (`disp` >= 1.7692467 AND `wt` >= -0.978325605) THEN 0.929919958
END) + CASE
WHEN ((`wt` < -0.978325605 OR (`wt` IS NULL))) THEN 3.34957242
WHEN ((`hp` < 0.668182373 OR (`hp` IS NULL)) AND `wt` >= -0.978325605) THEN 1.93179655
WHEN (`hp` >= 0.668182373 AND `wt` >= -0.978325605) THEN 1.05857491
END) + CASE
WHEN ((`disp` < -1.04222393 OR (`disp` IS NULL))) THEN 2.62246895
WHEN ((`hp` < 0.668182373 OR (`hp` IS NULL)) AND `disp` >= -1.04222393) THEN 1.39560235
WHEN (`hp` >= 0.668182373 AND `disp` >= -1.04222393) THEN 0.780699074
END) + CASE
WHEN (`wt` >= 0.110122316) THEN 0.68489337
WHEN ((`disp` < -1.04222393 OR (`disp` IS NULL)) AND (`wt` < 0.110122316 OR (`wt` IS NULL))) THEN 1.96685159
WHEN (`disp` >= -1.04222393 AND (`wt` < 0.110122316 OR (`wt` IS NULL))) THEN 1.15808082
END) + CASE
WHEN ((`disp` < -1.04222393 OR (`disp` IS NULL))) THEN 1.47513843
WHEN ((`wt` < 0.110122316 OR (`wt` IS NULL)) AND `disp` >= -1.04222393) THEN 0.83960855
WHEN ((`qsec` < -0.0776465684 OR (`qsec` IS NULL)) AND `wt` >= 0.110122316 AND `disp` >= -1.04222393) THEN 0.567495942
WHEN (`qsec` >= -0.0776465684 AND `wt` >= 0.110122316 AND `disp` >= -1.04222393) THEN 0.301891834
END) + CASE
WHEN ((`disp` < -1.22537899 OR (`disp` IS NULL))) THEN 1.30021882
WHEN ((`drat` < -1.26536262 OR (`drat` IS NULL)) AND (`hp` < 0.449405074 OR (`hp` IS NULL)) AND `disp` >= -1.22537899) THEN 0.119869716
WHEN (`drat` >= -1.26536262 AND (`hp` < 0.449405074 OR (`hp` IS NULL)) AND `disp` >= -1.22537899) THEN 0.631567359
WHEN ((`wt` < 0.549589574 OR (`wt` IS NULL)) AND `hp` >= 0.449405074 AND `disp` >= -1.22537899) THEN 0.35708189
WHEN (`wt` >= 0.549589574 AND `hp` >= 0.449405074 AND `disp` >= -1.22537899) THEN 0.191202134
END) + CASE
WHEN ((`disp` < -1.22537899 OR (`disp` IS NULL))) THEN 1.00766969
WHEN ((`qsec` < -0.377040505 OR (`qsec` IS NULL)) AND (`hp` < 0.668182373 OR (`hp` IS NULL)) AND `disp` >= -1.22537899) THEN 0.549484432
WHEN (`qsec` >= -0.377040505 AND (`hp` < 0.668182373 OR (`hp` IS NULL)) AND `disp` >= -1.22537899) THEN 0.298643202
WHEN ((`disp` < 0.966430426 OR (`disp` IS NULL)) AND `hp` >= 0.668182373 AND `disp` >= -1.22537899) THEN 0.0644110739
WHEN (`disp` >= 0.966430426 AND `hp` >= 0.668182373 AND `disp` >= -1.22537899) THEN 0.194638208
END) + CASE
WHEN ((`disp` < -1.22537899 OR (`disp` IS NULL))) THEN 0.78094399
WHEN ((`drat` < 0.754541874 OR (`drat` IS NULL)) AND (`hp` < -0.418411613 OR (`hp` IS NULL)) AND `disp` >= -1.22537899) THEN 0.430569261
WHEN (`drat` >= 0.754541874 AND (`hp` < -0.418411613 OR (`hp` IS NULL)) AND `disp` >= -1.22537899) THEN 0.0782236606
WHEN ((`hp` < 0.230627745 OR (`hp` IS NULL)) AND `hp` >= -0.418411613 AND `disp` >= -1.22537899) THEN (-0.0337554961)
WHEN (`hp` >= 0.230627745 AND `hp` >= -0.418411613 AND `disp` >= -1.22537899) THEN 0.245185137
END) + 0.5 AS `prediction`,
  `am`,
  `carb`,
  `vs`,
  `qsec`,
  `wt`,
  `drat`,
  `disp`,
  `hp`,
  `cyl`,
  `gear`
FROM (
  SELECT
    `am`,
    `carb`,
    `vs`,
    (`qsec` - 17.84875) / 1.78694323609684 AS `qsec`,
    (`wt` - 3.21725) / 0.978457442989697 AS `wt`,
    (`drat` - 3.5965625) / 0.534678736070971 AS `drat`,
    (`disp` - 230.721875) / 123.938693831382 AS `disp`,
    (`hp` - 146.6875) / 68.5628684893206 AS `hp`,
    (`cyl` - 6.1875) / 1.78592164694654 AS `cyl`,
    `gear`,
    `mpg`
  FROM churn2.test
) `q01`
ORDER BY `mpg`