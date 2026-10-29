-- chipsynth C64 dataset generator (Osc1 enum wave + optional PW + pre-roll)
-- ReaScript (Lua) for REAPER

-- ====== USER CONFIG ======
local OUT_DIR       = [[/Users/hayakzan/Desktop/JUCE/NEW PROJ/data]]
local NOTE_LEN      = 2.0
local SILENCE_LEAD  = 0.20
local VEL_LEVELS    = {80, 110}
local MIDI_NOTES    = {33, 45, 57, 69, 81, 86, 91}
-- local WAVES         = {"triangle"}  -- choose subset of waves
local WAVES         = {"noise", "saw", "pulse", "triangle"}  -- choose subset of waves


-- Osc 1 Wave enum (you probed these)
local ENUM_WAVE_PARAM = 7
local ENUM_WAVE_VALUES = {
  noise    = 0.000000,
  saw      = 0.333333,
  pulse    = 0.666667,
  triangle = 1.000000
}

-- Optional: Osc 1 Pulse Width param index (set if you find it)
local PW_PARAM_IDX = -1
local PW_VALUES    = {0.10, 0.30, 0.50, 0.70, 0.90}

-- ====== UTIL ======
local r = reaper
local function log(s) r.ShowConsoleMsg(tostring(s).."\n") end

local function ensure_dir_exists(path)
  r.RecursiveCreateDirectory(path, 0) -- ok if exists
  local probe = path .. "/.write_test"
  local f = io.open(probe, "w")
  if not f then log("ERROR: cannot write to OUT_DIR: "..path); return false end
  f:write("ok"); f:close(); os.remove(probe); return true
end

local function freq_of_midi(n) return 440.0 * (2 ^ ((n - 69)/12.0)) end
local function get_track(idx) return r.GetTrack(0, idx) end
local function find_fx_on_track(tr, needle)
  for i=0,r.TrackFX_GetCount(tr)-1 do
    local _, name = r.TrackFX_GetFXName(tr,i,"")
    if name:lower():find(needle,1,true) then return i, name end
  end
  return -1,nil
end

local function set_enum_wave(tr, fx, wave)
  local v = ENUM_WAVE_VALUES[wave]; if not v then error("Unknown wave: "..tostring(wave)) end
  r.TrackFX_SetParamNormalized(tr, fx, ENUM_WAVE_PARAM, v)
  r.TrackFX_EndParamEdit(tr, fx, ENUM_WAVE_PARAM)
end

local function set_pw(tr, fx, pw_norm)
  if PW_PARAM_IDX and PW_PARAM_IDX >= 0 then
    r.TrackFX_SetParamNormalized(tr, fx, PW_PARAM_IDX, math.max(0, math.min(1, pw_norm)))
    r.TrackFX_EndParamEdit(tr, fx, PW_PARAM_IDX)
  end
end

local function make_fresh_midi_item(track, t_end, t0, t1, pitch, vel)
  -- clear previous items so we always have a real MIDI item
  for i=r.CountTrackMediaItems(track)-1,0,-1 do
    r.DeleteTrackMediaItem(track, r.GetTrackMediaItem(track,i))
  end
  local item = r.CreateNewMIDIItemInProj(track, 0, t_end, false)
  local take = r.GetActiveTake(item)
  local ppq0 = r.MIDI_GetPPQPosFromProjTime(take, t0)
  local ppq1 = r.MIDI_GetPPQPosFromProjTime(take, t1)
  r.MIDI_InsertNote(take, false,false, ppq0, ppq1, 0, pitch, vel, false)
  r.MIDI_Sort(take)
  return item, take
end

-- ====== MAIN ======
r.Undo_BeginBlock(); r.ShowConsoleMsg("")
if not ensure_dir_exists(OUT_DIR) then return end

local track = get_track(0); if not track then log("ERROR: No Track 1."); return end
local fx_idx, fx_name = find_fx_on_track(track, "chipsynth"); if fx_idx<0 then log("ERROR: chipsynth not found."); return end
log("Using FX: "..fx_name)

local waves = {"noise", "saw", "pulse", "triangle"}
-- local waves = {"noise", "saw", "pulse", "triangle"}
local manifest_path = OUT_DIR.."/manifest.csv"
local mf = io.open(manifest_path,"w"); mf:write("file,wave,midi_note,freq_hz,velocity,pw_norm\n")

for _, wave in ipairs(waves) do
  set_enum_wave(track, fx_idx, wave)

  for _, midi in ipairs(MIDI_NOTES) do
    local hz = freq_of_midi(midi)
    for _, vel in ipairs(VEL_LEVELS) do
      local pw_list = (wave=="pulse" and PW_VALUES) or {nil}
      for _, pw in ipairs(pw_list) do
        if pw then set_pw(track, fx_idx, pw) end

        local t0 = SILENCE_LEAD
        local t1 = SILENCE_LEAD + NOTE_LEN
        local t_end = t1
        r.GetSet_LoopTimeRange(true,false, 0, t_end, false) -- render window
        r.Main_OnCommand(40345,0) -- all notes off

        -- create a real MIDI item and insert the note
        make_fresh_midi_item(track, t_end, t0, t1, midi, vel)

        local pw_tag = (pw and string.format("_pw%02d", math.floor(pw*100))) or ""
        local fname = string.format("%s_%s_m%03d_%0.1fHz_vel%d%s",
                        wave, "chipsynth", midi, hz, vel, pw_tag)
        local full_no_ext = OUT_DIR.."/"..fname
        r.GetSetProjectInfo_String(0,"RENDER_FILE",full_no_ext,true)
        r.Main_OnCommand(41824,0) -- render

        mf:write(string.format("%s.wav,%s,%d,%.4f,%d,%s\n",
          fname, wave, midi, hz, vel, pw and string.format("%.3f", pw) or ""))

        r.UpdateArrange()
      end
    end
  end
end

mf:close()
r.Undo_EndBlock("Generate chipsynth C64 dataset (Osc1 enum wave)", -1)
log("Done. Files and manifest written to: "..OUT_DIR)

