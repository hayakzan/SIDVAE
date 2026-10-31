-- print_osc1.lua
local r = reaper
r.ShowConsoleMsg("print_osc1 running\n")

local n = r.CountTracks(0)
r.ShowConsoleMsg("tracks in project: " .. n .. "\n")

for t = 0, n - 1 do
  local tr = r.GetTrack(0, t)
  for fx = 0, r.TrackFX_GetCount(tr) - 1 do
    local _, name = r.TrackFX_GetFXName(tr, fx, "")
    r.ShowConsoleMsg(string.format("track %d, fx %d: %s\n", t + 1, fx, name))
    if name:lower():find("chipsynth", 1, true) then
      for _, p in ipairs({5, 7}) do
        local _, pname = r.TrackFX_GetParamName(tr, fx, p, "")
        local _, shown = r.TrackFX_GetFormattedParamValue(tr, fx, p, "")
        r.ShowConsoleMsg(string.format("  [%d] %-28s normalized %.6f  shown %s\n",
          p, pname, r.TrackFX_GetParamNormalized(tr, fx, p), shown))
      end
    end
  end
end
r.ShowConsoleMsg("done\n")
